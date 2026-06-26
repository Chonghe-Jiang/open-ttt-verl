from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any
from uuid import uuid4

from guidance_ttt.puct import rank_archive_nodes
from guidance_ttt.state import LibraryEntry, LibraryNode


class GuidanceLibrary:
    """JSON-backed PUCT library for guidance + execution TTT."""

    def __init__(
        self,
        path: str | Path,
        *,
        initial_nodes: list[LibraryNode] | None = None,
        rollout_n: int = 1,
        puct_c: float = 1.0,
        max_buffer_size: int = 1000,
        topk_children: int = 2,
    ) -> None:
        self.path = Path(path)
        self.rollout_n = int(rollout_n)
        self.puct_c = float(puct_c)
        self.max_buffer_size = int(max_buffer_size)
        self.topk_children = int(topk_children)
        self._thread_lock = threading.RLock()
        self._nodes: dict[str, LibraryNode] = {}
        self._entries: dict[str, LibraryEntry] = {}
        self._groups: dict[str, dict[str, Any]] = {}
        self._best_node_id: str | None = None
        self._puct_n: dict[str, int] = {}
        self._puct_m: dict[str, float] = {}
        self._puct_T: int = 0

        if self.path.exists():
            self._load()
        else:
            for node in initial_nodes or []:
                self._nodes[node.id] = node
            self._refresh_best()
            self._save()

    @contextmanager
    def _file_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w") as lock_file:
            flock(lock_file, LOCK_EX)
            try:
                yield
            finally:
                flock(lock_file, LOCK_UN)

    def snapshot(self) -> dict[str, Any]:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                return self._to_store()

    def acquire_group(self, group_uid: str, *, visible_timestep_exclusive: int | None = None) -> LibraryNode:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                group = self._groups.get(group_uid)
                if group is not None:
                    return self._nodes[group["selected_node_id"]]
                selected = self._select_node(
                    visible_timestep_exclusive=visible_timestep_exclusive,
                    blocked_node_ids=self._same_step_blocked_node_ids(visible_timestep_exclusive),
                )
                selected.visits += 1
                self._groups[group_uid] = {
                    "selected_node_id": selected.id,
                    "submitted": 0,
                    "children": [],
                    "finalized": False,
                    "visible_timestep_exclusive": visible_timestep_exclusive,
                }
                self._save()
                return selected

    def submit_child(self, group_uid: str, entry: LibraryEntry) -> LibraryNode:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                if group_uid not in self._groups:
                    raise KeyError(f"Unknown group_uid: {group_uid}")
                group = self._groups[group_uid]
                parent_id = entry.parent_id if entry.parent_id in self._nodes else group["selected_node_id"]
                parent = self._nodes[parent_id]

                self._entries[entry.id] = entry
                child = LibraryNode(
                    id=str(uuid4()),
                    problem_id=entry.problem_id,
                    timestep=entry.timestep,
                    entry_id=entry.id,
                    value=entry.verifier_reward,
                    raw_score=entry.verifier_raw_score,
                    visits=0,
                    parent_id=parent.id,
                    children=[],
                    metadata={"verifier_status": entry.verifier_status},
                )
                self._nodes[child.id] = child
                parent.children.append(child.id)
                group["submitted"] += 1
                group["children"].append(child.id)
                if group["submitted"] >= self.rollout_n:
                    group["finalized"] = True
                    self._update_puct_stats_for_group(group)
                    self._filter_archive()
                    self._refresh_best()
                else:
                    self._refresh_best()
                self._save()
                return child

    def mark_node_visited(self, node_id: str, *, count: int) -> None:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                self._nodes[node_id].visits = int(count)
                self._puct_n[node_id] = int(count)
                self._save()

    def get_entry(self, entry_id: str | None) -> LibraryEntry | None:
        if entry_id is None:
            return None
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                return self._entries.get(entry_id)

    def context_for_node(
        self,
        node: LibraryNode,
        *,
        visible_timestep_exclusive: int | None = None,
    ) -> dict[str, Any]:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                lineage: list[LibraryEntry] = []
                current: LibraryNode | None = self._nodes.get(node.id)
                while current is not None:
                    if (
                        current.entry_id
                        and current.entry_id in self._entries
                        and self._entry_is_visible(
                            self._entries[current.entry_id],
                            visible_timestep_exclusive=visible_timestep_exclusive,
                        )
                    ):
                        lineage.append(self._entries[current.entry_id])
                    current = self._nodes.get(current.parent_id) if current.parent_id else None
                lineage.reverse()
                best = self._best_visible_node(visible_timestep_exclusive=visible_timestep_exclusive)
                best_entry = self._entries.get(best.entry_id) if best and best.entry_id else None
                failures = [
                    entry
                    for entry in self._entries.values()
                    if entry.parent_id == node.id and entry.verifier_status != "valid"
                    and self._entry_is_visible(entry, visible_timestep_exclusive=visible_timestep_exclusive)
                ]
                return {
                    "selected_entry": self._visible_entry(
                        node.entry_id,
                        visible_timestep_exclusive=visible_timestep_exclusive,
                    ),
                    "lineage_entries": lineage,
                    "global_best_entries": [best_entry] if best_entry else [],
                    "local_failure_entries": failures,
                }

    def _select_node(
        self,
        *,
        visible_timestep_exclusive: int | None = None,
        blocked_node_ids: set[str] | None = None,
    ) -> LibraryNode:
        visible_nodes = [
            node
            for node in self._nodes.values()
            if self._node_is_visible(node, visible_timestep_exclusive=visible_timestep_exclusive)
        ]
        if not visible_nodes:
            raise ValueError("GuidanceLibrary requires at least one root node")
        initial_ids = {node.id for node in visible_nodes if node.parent_id is None}
        ranked = rank_archive_nodes(
            visible_nodes,
            initial_ids=initial_ids,
            visit_counts=self._puct_n,
            best_reachable_values=self._puct_m,
            total_visits=self._puct_T,
            puct_c=self.puct_c,
        )
        blocked_node_ids = blocked_node_ids or set()
        for _score, _value, node, _n, _q, _prior, _bonus in ranked:
            if node.id not in blocked_node_ids:
                return node
        return ranked[0][2]

    def _same_step_blocked_node_ids(self, visible_timestep_exclusive: int | None) -> set[str]:
        if visible_timestep_exclusive is None:
            return set()
        selected_ids = {
            str(group["selected_node_id"])
            for group in self._groups.values()
            if group.get("visible_timestep_exclusive") == visible_timestep_exclusive
            and group.get("selected_node_id") in self._nodes
        }
        children_map = self._build_children_map()
        blocked: set[str] = set()
        for node_id in selected_ids:
            blocked.update(self._full_lineage_node_ids(node_id, children_map))
        return blocked

    def _ancestor_node_ids(self, node_id: str) -> list[str]:
        ancestors: list[str] = []
        current = self._nodes.get(node_id)
        while current is not None:
            ancestors.append(current.id)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        return ancestors

    def _build_children_map(self) -> dict[str, set[str]]:
        children_map: dict[str, set[str]] = {}
        for node in self._nodes.values():
            if node.parent_id:
                children_map.setdefault(node.parent_id, set()).add(node.id)
        return children_map

    def _full_lineage_node_ids(self, node_id: str, children_map: dict[str, set[str]]) -> set[str]:
        lineage = set(self._ancestor_node_ids(node_id))
        queue = [node_id]
        seen = {node_id}
        while queue:
            current_id = queue.pop(0)
            for child_id in children_map.get(current_id, set()):
                if child_id in seen:
                    continue
                seen.add(child_id)
                lineage.add(child_id)
                queue.append(child_id)
        return lineage

    def _update_puct_stats_for_group(self, group: dict[str, Any]) -> None:
        parent_max: dict[str, float] = {}
        for child_id in group.get("children", []):
            child = self._nodes.get(child_id)
            if child is None or child.parent_id is None:
                continue
            parent_max[child.parent_id] = max(parent_max.get(child.parent_id, float("-inf")), float(child.value))
        for parent_id, best_child_value in parent_max.items():
            self._puct_m[parent_id] = max(float(self._puct_m.get(parent_id, best_child_value)), best_child_value)
            for ancestor_id in self._ancestor_node_ids(parent_id):
                self._puct_n[ancestor_id] = int(self._puct_n.get(ancestor_id, 0)) + 1
            self._puct_T += 1

    def _node_construction_key(self, node: LibraryNode) -> str | None:
        if not node.entry_id:
            return None
        entry = self._entries.get(node.entry_id)
        if entry is None:
            return None
        artifacts = (entry.metadata or {}).get("verification_artifacts") or {}
        h_values = artifacts.get("h_values")
        if entry.verifier_status == "valid" and isinstance(h_values, list) and h_values:
            return json.dumps(
                {
                    "n_points": artifacts.get("n_points", len(h_values)),
                    "c5_bound": artifacts.get("c5_bound", entry.verifier_raw_score),
                    "h_values": h_values,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        return entry.summary or None

    def _filter_archive(self) -> None:
        keep_ids = self._topk_child_node_ids()
        keep_ids = self._dedup_node_ids(keep_ids)
        keep_ids = self._limit_buffer_node_ids(keep_ids)
        keep_ids = self._with_ancestor_closure(keep_ids)
        self._prune_nodes(keep_ids)

    def _topk_child_node_ids(self) -> set[str]:
        if self.topk_children <= 0:
            return set(self._nodes)
        keep_ids = {node.id for node in self._nodes.values() if node.parent_id is None}
        children_by_parent: dict[str, list[LibraryNode]] = {}
        for node in self._nodes.values():
            if node.parent_id is not None:
                children_by_parent.setdefault(node.parent_id, []).append(node)
        for children in children_by_parent.values():
            children.sort(key=lambda node: (node.value, node.id), reverse=True)
            keep_ids.update(child.id for child in children[: self.topk_children])
        return keep_ids

    def _dedup_node_ids(self, candidate_ids: set[str]) -> set[str]:
        roots = {node.id for node in self._nodes.values() if node.parent_id is None}
        sorted_nodes = sorted(
            (self._nodes[node_id] for node_id in candidate_ids if node_id in self._nodes and node_id not in roots),
            key=lambda node: (node.value, node.id),
            reverse=True,
        )
        keep_ids = set(roots)
        seen_keys: set[str] = set()
        for node in sorted_nodes:
            key = self._node_construction_key(node)
            if key is not None and key in seen_keys:
                continue
            keep_ids.add(node.id)
            if key is not None:
                seen_keys.add(key)
        return keep_ids

    def _limit_buffer_node_ids(self, candidate_ids: set[str]) -> set[str]:
        if self.max_buffer_size <= 0 or len(candidate_ids) <= self.max_buffer_size:
            return candidate_ids
        roots = {node.id for node in self._nodes.values() if node.parent_id is None}
        keep_ids = {node_id for node_id in roots if node_id in candidate_ids}
        sorted_nodes = sorted(
            (self._nodes[node_id] for node_id in candidate_ids if node_id in self._nodes and node_id not in keep_ids),
            key=lambda node: (node.value, node.id),
            reverse=True,
        )
        for node in sorted_nodes:
            if len(keep_ids) >= self.max_buffer_size:
                break
            keep_ids.add(node.id)
        return keep_ids

    def _with_ancestor_closure(self, candidate_ids: set[str]) -> set[str]:
        keep_ids = {node_id for node_id in candidate_ids if node_id in self._nodes}
        for node_id in list(keep_ids):
            keep_ids.update(self._ancestor_node_ids(node_id))
        return keep_ids

    def _prune_nodes(self, keep_ids: set[str]) -> None:
        if len(keep_ids) == len(self._nodes):
            return
        self._nodes = {node_id: node for node_id, node in self._nodes.items() if node_id in keep_ids}
        for node in self._nodes.values():
            node.children = [child_id for child_id in node.children if child_id in self._nodes]
        self._puct_n = {node_id: count for node_id, count in self._puct_n.items() if node_id in self._nodes}
        self._puct_m = {node_id: value for node_id, value in self._puct_m.items() if node_id in self._nodes}

    def _node_is_visible(self, node: LibraryNode, *, visible_timestep_exclusive: int | None) -> bool:
        if visible_timestep_exclusive is None:
            return True
        if node.parent_id is None:
            return True
        return node.timestep < int(visible_timestep_exclusive)

    def _entry_is_visible(self, entry: LibraryEntry, *, visible_timestep_exclusive: int | None) -> bool:
        if visible_timestep_exclusive is None:
            return True
        return entry.timestep < int(visible_timestep_exclusive)

    def _visible_entry(
        self,
        entry_id: str | None,
        *,
        visible_timestep_exclusive: int | None,
    ) -> LibraryEntry | None:
        if entry_id is None:
            return None
        entry = self._entries.get(entry_id)
        if entry is None or not self._entry_is_visible(entry, visible_timestep_exclusive=visible_timestep_exclusive):
            return None
        return entry

    def _best_visible_node(self, *, visible_timestep_exclusive: int | None) -> LibraryNode | None:
        visible_nodes = [
            node
            for node in self._nodes.values()
            if self._node_is_visible(node, visible_timestep_exclusive=visible_timestep_exclusive)
        ]
        if not visible_nodes:
            return None
        return max(visible_nodes, key=lambda node: (node.value, node.id))

    def _refresh_best(self) -> None:
        if not self._nodes:
            self._best_node_id = None
            return
        self._best_node_id = max(self._nodes.values(), key=lambda node: (node.value, node.id)).id

    def _to_store(self) -> dict[str, Any]:
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "entries": {entry_id: entry.to_dict() for entry_id, entry in self._entries.items()},
            "groups": self._groups,
            "best_node_id": self._best_node_id,
            "config": {
                "rollout_n": self.rollout_n,
                "puct_c": self.puct_c,
                "max_buffer_size": self.max_buffer_size,
                "topk_children": self.topk_children,
            },
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "puct_n": self._puct_n,
            "puct_m": self._puct_m,
            "puct_T": self._puct_T,
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._to_store(), indent=2, sort_keys=True))

    def _reload(self) -> None:
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self._nodes = {node_id: LibraryNode.from_dict(node) for node_id, node in data.get("nodes", {}).items()}
        self._entries = {
            entry_id: LibraryEntry.from_dict(entry) for entry_id, entry in data.get("entries", {}).items()
        }
        self._groups = dict(data.get("groups", {}))
        self._best_node_id = data.get("best_node_id")
        config = data.get("config", {})
        self.rollout_n = int(config.get("rollout_n", data.get("rollout_n", self.rollout_n)))
        self.puct_c = float(config.get("puct_c", data.get("puct_c", self.puct_c)))
        self.max_buffer_size = int(config.get("max_buffer_size", data.get("max_buffer_size", self.max_buffer_size)))
        self.topk_children = int(config.get("topk_children", data.get("topk_children", self.topk_children)))
        has_puct_n = "puct_n" in data
        self._puct_n = {str(node_id): int(count) for node_id, count in (data.get("puct_n") or {}).items()}
        self._puct_m = {str(node_id): float(value) for node_id, value in (data.get("puct_m") or {}).items()}
        self._puct_T = int(data.get("puct_T", 0) or 0)
        if not has_puct_n:
            self._puct_n = {node_id: int(node.visits) for node_id, node in self._nodes.items() if node.visits}
