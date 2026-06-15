from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any
from uuid import uuid4

from guidance_ttt.puct import choose_child
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
    ) -> None:
        self.path = Path(path)
        self.rollout_n = int(rollout_n)
        self.puct_c = float(puct_c)
        self._thread_lock = threading.RLock()
        self._nodes: dict[str, LibraryNode] = {}
        self._entries: dict[str, LibraryEntry] = {}
        self._groups: dict[str, dict[str, Any]] = {}
        self._best_node_id: str | None = None

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

    def acquire_group(self, group_uid: str) -> LibraryNode:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                group = self._groups.get(group_uid)
                if group is not None:
                    return self._nodes[group["selected_node_id"]]
                selected = self._select_node()
                selected.visits += 1
                self._groups[group_uid] = {
                    "selected_node_id": selected.id,
                    "submitted": 0,
                    "children": [],
                    "finalized": False,
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
                self._refresh_best()
                self._save()
                return child

    def mark_node_visited(self, node_id: str, *, count: int) -> None:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                self._nodes[node_id].visits = int(count)
                self._save()

    def get_entry(self, entry_id: str | None) -> LibraryEntry | None:
        if entry_id is None:
            return None
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                return self._entries.get(entry_id)

    def context_for_node(self, node: LibraryNode) -> dict[str, Any]:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                lineage: list[LibraryEntry] = []
                current: LibraryNode | None = self._nodes.get(node.id)
                while current is not None:
                    if current.entry_id and current.entry_id in self._entries:
                        lineage.append(self._entries[current.entry_id])
                    current = self._nodes.get(current.parent_id) if current.parent_id else None
                lineage.reverse()
                best = self._nodes.get(self._best_node_id) if self._best_node_id else None
                best_entry = self._entries.get(best.entry_id) if best and best.entry_id else None
                failures = [
                    entry
                    for entry in self._entries.values()
                    if entry.parent_id == node.id and entry.verifier_status != "valid"
                ]
                return {
                    "selected_entry": self._entries.get(node.entry_id) if node.entry_id else None,
                    "lineage_entries": lineage,
                    "global_best_entries": [best_entry] if best_entry else [],
                    "local_failure_entries": failures,
                }

    def _select_node(self) -> LibraryNode:
        roots = [node for node in self._nodes.values() if node.parent_id is None]
        if not roots:
            raise ValueError("GuidanceLibrary requires at least one root node")
        root = max(roots, key=lambda node: (node.value, node.id))
        if not root.children:
            return root
        children = [self._nodes[child_id] for child_id in root.children if child_id in self._nodes]
        return choose_child(root, children, puct_c=self.puct_c)

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
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
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
        self.rollout_n = int(data.get("rollout_n", self.rollout_n))
        self.puct_c = float(data.get("puct_c", self.puct_c))
