from __future__ import annotations

import hashlib
import json
import math
import threading
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any
from uuid import uuid4

from guidance_ttt.puct import PUCT_Q_BLEND, normalize_puct_q_mode, rank_archive_nodes
from guidance_ttt.state import LibraryEntry, LibraryNode
from guidance_ttt.strategy import strategy_frequency, strategy_similarity, strategy_tags


REFERENCE_SELECTION_PUCT_TOP2 = "puct_top2"
REFERENCE_SELECTION_DIVERSE_TOP2 = "diverse_top2"
SUPPORTED_REFERENCE_SELECTION_MODES = frozenset(
    {REFERENCE_SELECTION_PUCT_TOP2, REFERENCE_SELECTION_DIVERSE_TOP2}
)


def normalize_reference_selection_mode(mode: str | None) -> str:
    normalized = str(mode or REFERENCE_SELECTION_PUCT_TOP2).strip().lower()
    if normalized not in SUPPORTED_REFERENCE_SELECTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_REFERENCE_SELECTION_MODES))
        raise ValueError(f"Unsupported reference selection mode {mode!r}; expected one of: {supported}")
    return normalized


class GuidanceLibrary:
    """JSON-backed PUCT library for guidance + execution TTT."""

    def __init__(
        self,
        path: str | Path,
        *,
        initial_nodes: list[LibraryNode] | None = None,
        rollout_n: int = 1,
        puct_c: float = 1.0,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int = 1000,
        topk_children: int = 2,
        reference_selection_mode: str = REFERENCE_SELECTION_PUCT_TOP2,
    ) -> None:
        self.path = Path(path)
        self.rollout_n = int(rollout_n)
        self.puct_c = float(puct_c)
        self.puct_q_mode = normalize_puct_q_mode(puct_q_mode)
        self.max_buffer_size = int(max_buffer_size)
        self.topk_children = int(topk_children)
        self.reference_selection_mode = normalize_reference_selection_mode(reference_selection_mode)
        self._thread_lock = threading.RLock()
        self._nodes: dict[str, LibraryNode] = {}
        self._entries: dict[str, LibraryEntry] = {}
        self._groups: dict[str, dict[str, Any]] = {}
        self._best_node_id: str | None = None
        self._puct_n: dict[str, int] = {}
        self._puct_m: dict[str, float] = {}
        self._puct_T: int = 0

        with self._thread_lock:
            with self._file_lock():
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

    def configure_pristine_archive(
        self,
        *,
        rollout_n: int,
        puct_c: float,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int,
        topk_children: int,
        reference_selection_mode: str = REFERENCE_SELECTION_PUCT_TOP2,
    ) -> None:
        """Apply run-specific sampling config to an unused seed archive."""
        expected = self._coerce_runtime_config(
            rollout_n=rollout_n,
            puct_c=puct_c,
            puct_q_mode=puct_q_mode,
            max_buffer_size=max_buffer_size,
            topk_children=topk_children,
            reference_selection_mode=reference_selection_mode,
        )
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                if self._groups or self._puct_T or self._puct_n or self._puct_m:
                    raise ValueError(
                        f"Seed library {self.path} is not pristine: run-local groups or PUCT statistics "
                        "are already present. Use a clean bootstrap seed or a new output directory."
                    )
                self.rollout_n = expected["rollout_n"]
                self.puct_c = expected["puct_c"]
                self.puct_q_mode = expected["puct_q_mode"]
                self.max_buffer_size = expected["max_buffer_size"]
                self.topk_children = expected["topk_children"]
                self.reference_selection_mode = expected["reference_selection_mode"]
                self._save()

    def assert_runtime_config(
        self,
        *,
        rollout_n: int,
        puct_c: float,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int,
        topk_children: int,
        reference_selection_mode: str = REFERENCE_SELECTION_PUCT_TOP2,
    ) -> None:
        """Fail fast when a persisted archive disagrees with the active recipe."""
        expected = self._coerce_runtime_config(
            rollout_n=rollout_n,
            puct_c=puct_c,
            puct_q_mode=puct_q_mode,
            max_buffer_size=max_buffer_size,
            topk_children=topk_children,
            reference_selection_mode=reference_selection_mode,
        )
        actual = self._runtime_config()
        mismatches = []
        for key, expected_value in expected.items():
            actual_value = actual[key]
            matches = (
                math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12)
                if key == "puct_c"
                else actual_value == expected_value
            )
            if not matches:
                mismatches.append(f"{key}: archive={actual_value!r}, recipe={expected_value!r}")
        if mismatches:
            raise ValueError(
                f"Guidance library runtime config mismatch for {self.path}: "
                + "; ".join(mismatches)
                + ". Existing run libraries are immutable with respect to sampling config; "
                "use a new output directory."
            )
        self._assert_group_accounting()

    @staticmethod
    def _coerce_runtime_config(
        *,
        rollout_n: int,
        puct_c: float,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int,
        topk_children: int,
        reference_selection_mode: str = REFERENCE_SELECTION_PUCT_TOP2,
    ) -> dict[str, int | float | str]:
        config: dict[str, int | float | str] = {
            "rollout_n": int(rollout_n),
            "puct_c": float(puct_c),
            "puct_q_mode": normalize_puct_q_mode(puct_q_mode),
            "max_buffer_size": int(max_buffer_size),
            "topk_children": int(topk_children),
            "reference_selection_mode": normalize_reference_selection_mode(reference_selection_mode),
        }
        if config["rollout_n"] <= 0:
            raise ValueError(f"rollout_n must be positive, got {config['rollout_n']!r}")
        return config

    def _runtime_config(self) -> dict[str, int | float | str]:
        return {
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "puct_q_mode": self.puct_q_mode,
            "max_buffer_size": self.max_buffer_size,
            "topk_children": self.topk_children,
            "reference_selection_mode": self.reference_selection_mode,
        }

    def _assert_group_accounting(self) -> None:
        errors: list[str] = []
        finalized_count = 0
        for group_uid, group in self._groups.items():
            submitted = int(group.get("submitted", 0))
            finalized = bool(group.get("finalized", False))
            if submitted < 0 or submitted > self.rollout_n:
                errors.append(
                    f"group {group_uid!r} has submitted={submitted}, expected 0..{self.rollout_n}"
                )
            if finalized:
                finalized_count += 1
                if submitted != self.rollout_n:
                    errors.append(
                        f"group {group_uid!r} is finalized with submitted={submitted}, "
                        f"expected {self.rollout_n}"
                    )
            elif submitted >= self.rollout_n:
                errors.append(
                    f"group {group_uid!r} is not finalized with submitted={submitted}, "
                    f"expected less than {self.rollout_n}"
                )
        if self._puct_T != finalized_count:
            errors.append(f"puct_T={self._puct_T}, expected one update for each of {finalized_count} finalized groups")
        if errors:
            raise ValueError(f"Guidance library group accounting is inconsistent for {self.path}: " + "; ".join(errors))

    def acquire_group(
        self,
        group_uid: str,
        *,
        visible_timestep_exclusive: int | None = None,
        require_solution: bool = False,
    ) -> LibraryNode:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                group = self._groups.get(group_uid)
                if group is not None:
                    selected = self._nodes[group["selected_node_id"]]
                    if require_solution and not self._node_has_solution(selected):
                        raise RuntimeError(
                            f"Existing group {group_uid!r} is attached to node {selected.id!r}, which has no "
                            "solution code. Start a new run from a code-bearing bootstrap library."
                        )
                    return selected
                blocked_node_ids = self._same_step_blocked_node_ids(visible_timestep_exclusive)
                ranked = self._rank_nodes(
                    visible_timestep_exclusive=visible_timestep_exclusive,
                    require_solution=require_solution,
                )
                selected = self._first_unblocked_node(ranked, blocked_node_ids)
                # Keep the alternatives fixed for the life of a group.  They are
                # ranked by exactly the same PUCT calculation as the main parent;
                # only the main parent is subject to same-step subtree blocking.
                reference_nodes = self._reference_nodes_for_selection(ranked, selected)
                reference_node_ids = [node.id for node in reference_nodes]
                selected.visits += 1
                self._groups[group_uid] = {
                    "selected_node_id": selected.id,
                    "reference_node_ids": reference_node_ids,
                    "reference_selection": self._reference_selection_metadata(selected, reference_nodes),
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
                submitted = int(group.get("submitted", 0))
                if bool(group.get("finalized", False)) or submitted >= self.rollout_n:
                    raise RuntimeError(
                        f"Group {group_uid!r} is already complete: submitted={submitted}, "
                        f"rollout_n={self.rollout_n}. Refusing an extra child submission."
                    )
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
                group["submitted"] = submitted + 1
                group["children"].append(child.id)
                if group["submitted"] == self.rollout_n:
                    group["finalized"] = True
                    self._update_puct_stats_for_group(group)
                    self._filter_archive()
                    self._refresh_best()
                else:
                    self._refresh_best()
                self._save()
                return child

    def attach_entry_to_root(self, root_node_id: str, entry: LibraryEntry, *, overwrite_existing: bool = False) -> None:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                if root_node_id not in self._nodes:
                    raise KeyError(f"Unknown root node id: {root_node_id}")
                root = self._nodes[root_node_id]
                if root.parent_id is not None:
                    raise ValueError(f"Node {root_node_id} is not a root node")
                if root.entry_id and not overwrite_existing:
                    raise ValueError(f"Root node {root_node_id} already has entry {root.entry_id}")

                entry.parent_id = root.id
                entry.timestep = 0
                self._entries[entry.id] = entry
                root.entry_id = entry.id
                root.value = entry.verifier_reward
                root.raw_score = entry.verifier_raw_score
                root.metadata.update(
                    {
                        "bootstrap": bool((entry.metadata or {}).get("bootstrap")),
                        "verifier_status": entry.verifier_status,
                    }
                )
                self._refresh_best()
                self._save()

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
        reference_node_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                selected_node = self._nodes.get(node.id)
                lineage: list[LibraryEntry] = []
                current: LibraryNode | None = selected_node
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
                visible_entries = [
                    entry
                    for entry in self._entries.values()
                    if self._entry_is_visible(entry, visible_timestep_exclusive=visible_timestep_exclusive)
                ]
                recent_entries = sorted(visible_entries, key=lambda entry: (entry.timestep, entry.id), reverse=True)[:12]
                invalid_entries = [entry for entry in visible_entries if entry.verifier_status != "valid"]
                previous_parent = (
                    self._nodes.get(selected_node.parent_id)
                    if selected_node and selected_node.parent_id
                    else None
                )
                reference_entries: list[LibraryEntry] = []
                for node_id in reference_node_ids or []:
                    reference_node = self._nodes.get(node_id)
                    reference_entry = self._visible_entry(
                        reference_node.entry_id if reference_node is not None else None,
                        visible_timestep_exclusive=visible_timestep_exclusive,
                    )
                    if reference_entry is not None:
                        reference_entries.append(reference_entry)
                return {
                    "selected_entry": self._visible_entry(
                        selected_node.entry_id if selected_node is not None else None,
                        visible_timestep_exclusive=visible_timestep_exclusive,
                    ),
                    "lineage_entries": lineage,
                    "previous_parent_entry": self._visible_entry(
                        previous_parent.entry_id if previous_parent is not None else None,
                        visible_timestep_exclusive=visible_timestep_exclusive,
                    ),
                    "reference_entries": reference_entries,
                    "global_best_entries": [best_entry] if best_entry else [],
                    "local_failure_entries": failures,
                    "search_history": {
                        "strategy_frequency": strategy_frequency(entry.guidance for entry in visible_entries),
                        "failure_strategy_frequency": strategy_frequency(entry.guidance for entry in invalid_entries),
                        "recent_strategies": [
                            {
                                "timestep": entry.timestep,
                                "tags": strategy_tags(entry.guidance),
                                "score": entry.verifier_raw_score,
                                "status": entry.verifier_status,
                            }
                            for entry in recent_entries
                        ],
                    },
                }

    def reference_nodes_for_group(
        self,
        group_uid: str,
        *,
        visible_timestep_exclusive: int | None = None,
    ) -> list[LibraryNode]:
        """Return the PUCT-ranked alternatives bound when this group was acquired."""
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                group = self._groups.get(group_uid)
                if group is None:
                    raise KeyError(f"Unknown group_uid: {group_uid}")
                return [
                    node
                    for node_id in group.get("reference_node_ids", [])
                    if (node := self._nodes.get(str(node_id))) is not None
                    and self._node_is_visible(node, visible_timestep_exclusive=visible_timestep_exclusive)
                ]

    def group_metadata(self, group_uid: str) -> dict[str, Any]:
        """Return immutable sampling metadata recorded when a group was acquired."""
        with self._thread_lock:
            with self._file_lock():
                self._reload()
                group = self._groups.get(group_uid)
                if group is None:
                    raise KeyError(f"Unknown group_uid: {group_uid}")
                return dict(group.get("reference_selection") or {})

    def _reference_nodes_for_selection(
        self,
        ranked: list[tuple[float, float, LibraryNode, int, float, float, float]],
        selected: LibraryNode,
    ) -> list[LibraryNode]:
        candidates = [node for _score, _value, node, *_rest in ranked if node.id != selected.id]
        if self.reference_selection_mode == REFERENCE_SELECTION_PUCT_TOP2:
            return candidates[:2]
        if not candidates:
            return []

        first = candidates[0]
        # Search only the leading PUCT candidates, then prefer an unrelated branch
        # and the lowest strategy-text similarity. This keeps the second reference
        # high-quality while breaking the common top-2 family collapse.
        pool = candidates[1 : min(len(candidates), 16)]
        if not pool:
            return [first]
        selected_text = self._node_strategy_text(selected)
        first_text = self._node_strategy_text(first)
        selected_branch = self._strategy_branch_id(selected)
        first_branch = self._strategy_branch_id(first)

        def key(item: tuple[int, LibraryNode]) -> tuple[int, int, float, int]:
            rank, candidate = item
            candidate_branch = self._strategy_branch_id(candidate)
            branch_diversity = int(candidate_branch != selected_branch) + int(candidate_branch != first_branch)
            unrelated = int(
                not self._nodes_are_lineage_related(candidate.id, selected.id)
                and not self._nodes_are_lineage_related(candidate.id, first.id)
            )
            similarity = max(
                strategy_similarity(self._node_strategy_text(candidate), selected_text),
                strategy_similarity(self._node_strategy_text(candidate), first_text),
            )
            return branch_diversity, unrelated, -similarity, -rank

        _, second = max(enumerate(pool, start=1), key=key)
        return [first, second]

    def _node_strategy_text(self, node: LibraryNode) -> str:
        entry = self._entries.get(node.entry_id) if node.entry_id else None
        if entry is None:
            return ""
        raw_summary = (entry.metadata or {}).get("raw_model_summary")
        return "\n".join(
            str(part)
            for part in (entry.guidance, raw_summary, entry.summary, entry.reusable_idea)
            if isinstance(part, str) and part.strip()
        )

    def _strategy_branch_id(self, node: LibraryNode) -> str:
        lineage = list(reversed(self._ancestor_node_ids(node.id)))
        return lineage[1] if len(lineage) > 1 else lineage[0]

    def _nodes_are_lineage_related(self, left_id: str, right_id: str) -> bool:
        return left_id in self._ancestor_node_ids(right_id) or right_id in self._ancestor_node_ids(left_id)

    def _reference_selection_metadata(
        self,
        selected: LibraryNode,
        references: list[LibraryNode],
    ) -> dict[str, Any]:
        reference_texts = [self._node_strategy_text(node) for node in references]
        return {
            "mode": self.reference_selection_mode,
            "selected_branch_id": self._strategy_branch_id(selected),
            "reference_node_ids": [node.id for node in references],
            "reference_branch_ids": [self._strategy_branch_id(node) for node in references],
            "reference_strategy_similarity": (
                strategy_similarity(reference_texts[0], reference_texts[1]) if len(reference_texts) == 2 else None
            ),
            "reference_lineage_related": (
                self._nodes_are_lineage_related(references[0].id, references[1].id) if len(references) == 2 else None
            ),
        }

    def _rank_nodes(
        self,
        *,
        visible_timestep_exclusive: int | None = None,
        require_solution: bool = False,
    ) -> list[tuple[float, float, LibraryNode, int, float, float, float]]:
        visible_nodes = [
            node
            for node in self._nodes.values()
            if self._node_is_visible(node, visible_timestep_exclusive=visible_timestep_exclusive)
            and (not require_solution or self._node_has_solution(node))
        ]
        if not visible_nodes:
            if require_solution:
                raise RuntimeError(
                    "GuidanceLibrary has no visible node with solution code. "
                    "Initialize the run from a valid code-bearing bootstrap library."
                )
            raise ValueError("GuidanceLibrary requires at least one root node")
        initial_ids = {node.id for node in visible_nodes if node.parent_id is None}
        return rank_archive_nodes(
            visible_nodes,
            initial_ids=initial_ids,
            visit_counts=self._puct_n,
            best_reachable_values=self._puct_m,
            total_visits=self._puct_T,
            puct_c=self.puct_c,
            q_mode=self.puct_q_mode,
        )

    @staticmethod
    def _first_unblocked_node(
        ranked: list[tuple[float, float, LibraryNode, int, float, float, float]],
        blocked_node_ids: set[str] | None,
    ) -> LibraryNode:
        blocked_node_ids = blocked_node_ids or set()
        for _score, _value, node, _n, _q, _prior, _bonus in ranked:
            if node.id not in blocked_node_ids:
                return node
        return ranked[0][2]

    def _select_node(
        self,
        *,
        visible_timestep_exclusive: int | None = None,
        blocked_node_ids: set[str] | None = None,
        require_solution: bool = False,
    ) -> LibraryNode:
        return self._first_unblocked_node(
            self._rank_nodes(
                visible_timestep_exclusive=visible_timestep_exclusive,
                require_solution=require_solution,
            ),
            blocked_node_ids,
        )

    def _node_has_solution(self, node: LibraryNode) -> bool:
        if not node.entry_id:
            return False
        entry = self._entries.get(node.entry_id)
        return entry is not None and bool(entry.solution.strip())

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
        solution = entry.solution.strip()
        if solution:
            digest = hashlib.sha256(solution.encode("utf-8")).hexdigest()
            return f"solution-sha256:{digest}"
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
        config = {
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "max_buffer_size": self.max_buffer_size,
            "topk_children": self.topk_children,
        }
        if self.puct_q_mode != PUCT_Q_BLEND:
            config["puct_q_mode"] = self.puct_q_mode
        if self.reference_selection_mode != REFERENCE_SELECTION_PUCT_TOP2:
            config["reference_selection_mode"] = self.reference_selection_mode
        store = {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "entries": {entry_id: entry.to_dict() for entry_id, entry in self._entries.items()},
            "groups": self._groups,
            "best_node_id": self._best_node_id,
            "config": config,
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "puct_n": self._puct_n,
            "puct_m": self._puct_m,
            "puct_T": self._puct_T,
        }
        if self.puct_q_mode != PUCT_Q_BLEND:
            store["puct_q_mode"] = self.puct_q_mode
        return store

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(json.dumps(self._to_store(), indent=2, sort_keys=True))
            temp_path.replace(self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

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
        self.puct_q_mode = normalize_puct_q_mode(
            config.get("puct_q_mode", data.get("puct_q_mode", self.puct_q_mode))
        )
        self.max_buffer_size = int(config.get("max_buffer_size", data.get("max_buffer_size", self.max_buffer_size)))
        self.topk_children = int(config.get("topk_children", data.get("topk_children", self.topk_children)))
        self.reference_selection_mode = normalize_reference_selection_mode(
            config.get("reference_selection_mode", self.reference_selection_mode)
        )
        has_puct_n = "puct_n" in data
        self._puct_n = {str(node_id): int(count) for node_id, count in (data.get("puct_n") or {}).items()}
        self._puct_m = {str(node_id): float(value) for node_id, value in (data.get("puct_m") or {}).items()}
        self._puct_T = int(data.get("puct_T", 0) or 0)
        if not has_puct_n:
            self._puct_n = {node_id: int(node.visits) for node_id, node in self._nodes.items() if node.visits}
