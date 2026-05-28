from __future__ import annotations

import json
import math
import os
import threading
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any

import numpy as np

from verl_ttt_discover.state import DiscoveryState


class PUCTArchive:
    """Persistent state archive with atomic group_uid -> state_id binding."""

    def __init__(
        self,
        path: str | Path,
        *,
        initial_states: list[DiscoveryState] | None = None,
        rollout_n: int = 1,
        puct_c: float = 1.0,
        topk_children: int = 2,
        max_buffer_size: int = 1000,
        max_construction_len: int | None = 1000,
    ) -> None:
        self.path = Path(path)
        self.rollout_n = int(rollout_n)
        self.puct_c = float(puct_c)
        self.topk_children = int(topk_children)
        self.max_buffer_size = int(max_buffer_size)
        self.max_construction_len = max_construction_len
        self._lock = threading.RLock()

        self._states: list[DiscoveryState] = []
        self._groups: dict[str, dict[str, Any]] = {}
        self._puct_n: dict[str, int] = {}
        self._puct_m: dict[str, float] = {}
        self._puct_T = 0
        self._best_state_id: str | None = None
        self._last_sampled_stats: list[dict[str, Any]] = []

        if self.path.exists():
            self._load()
        elif initial_states:
            self._states = list(initial_states)
            self._refresh_best()
            self._save()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            with self._file_lock():
                self._reload_from_disk()
                return self._to_store()

    def acquire_group(self, group_uid: str) -> DiscoveryState:
        with self._lock:
            with self._file_lock():
                self._reload_from_disk()
                group = self._groups.get(group_uid)
                if group is not None:
                    return self._state_by_id(group["state_id"])

                blocked_ids = self._blocked_ids_for_group(group_uid)
                state, puct_stats = self._sample_state(blocked_ids=blocked_ids)
                self._groups[group_uid] = {
                    "state_id": state.id,
                    "children": [],
                    "submitted": 0,
                    "finalized": False,
                    "puct_stats": puct_stats,
                }
                self._last_sampled_stats = [puct_stats]
                self._save()
                return state

    def submit_child(self, group_uid: str, child: DiscoveryState | None) -> bool:
        """Submit one rollout result. Returns True only when this call finalizes the group."""
        with self._lock:
            with self._file_lock():
                self._reload_from_disk()
                if group_uid not in self._groups:
                    raise KeyError(f"Unknown group_uid: {group_uid}")
                group = self._groups[group_uid]
                if group["finalized"]:
                    return False

                group["submitted"] += 1
                if child is not None and child.value is not None:
                    parent = self._state_by_id(group["state_id"])
                    self._set_parent(child, parent)
                    group["children"].append(child.to_dict())

                finalized = group["submitted"] >= self.rollout_n
                snapshot_step = None
                if finalized:
                    self._finalize_group(group_uid)
                    snapshot_step = _step_from_group_uid(group_uid)
                self._save(snapshot_step=snapshot_step)
                return finalized

    def _finalize_group(self, group_uid: str) -> None:
        group = self._groups[group_uid]
        if group["finalized"]:
            return

        parent = self._state_by_id(group["state_id"])
        children = [DiscoveryState.from_dict(data) for data in group["children"]]
        children.sort(key=lambda state: float(state.value), reverse=True)
        kept_children = children[: self.topk_children] if self.topk_children > 0 else children

        if children:
            best_value = float(children[0].value)
            self._puct_m[parent.id] = max(self._puct_m.get(parent.id, best_value), best_value)
            self._add_states(kept_children)

        ancestor_ids = [parent.id] + [str(p["id"]) for p in parent.parents if p.get("id")]
        for state_id in ancestor_ids:
            self._puct_n[state_id] = self._puct_n.get(state_id, 0) + 1
        self._puct_T += 1
        group["finalized"] = True
        self._refresh_best()

    def _sample_state(self, *, blocked_ids: set[str] | None = None) -> tuple[DiscoveryState, dict[str, Any]]:
        if not self._states:
            raise ValueError("PUCTArchive has no states to sample")
        values = np.array([float(state.value) for state in self._states], dtype=np.float64)
        scale = max(float(values.max() - values.min()), 1e-6)
        ranks = np.argsort(np.argsort(-values))
        weights = (len(values) - ranks).astype(np.float64)
        priors = weights / weights.sum()
        sqrt_T = math.sqrt(1.0 + self._puct_T)

        scored = []
        for idx, state in enumerate(self._states):
            visits = self._puct_n.get(state.id, 0)
            q_value = self._puct_m.get(state.id, float(state.value)) if visits > 0 else float(state.value)
            bonus = self.puct_c * scale * float(priors[idx]) * sqrt_T / (1.0 + visits)
            score = q_value + bonus
            scored.append((score, float(state.value), state, visits, q_value, float(priors[idx]), bonus))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        picked = None
        for entry in scored:
            if not blocked_ids or entry[2].id not in blocked_ids:
                picked = entry
                break
        if picked is None:
            picked = scored[0]
        score, value, state, visits, q_value, prior, bonus = picked
        return state, {
            "state_id": state.id,
            "timestep": state.timestep,
            "value": value,
            "construction_len": len(state.construction) if state.construction else 0,
            "n": visits,
            "Q": q_value,
            "P": prior,
            "bonus": bonus,
            "score": score,
            "blocked_candidates": len(blocked_ids or ()),
        }

    def _add_states(self, states: list[DiscoveryState]) -> None:
        existing_ids = {state.id for state in self._states}
        existing_keys = {key for state in self._states if (key := self._state_key(state)) is not None}
        for state in states:
            key = self._state_key(state)
            if state.id in existing_ids or (key is not None and key in existing_keys):
                continue
            if self.max_construction_len is not None and state.construction and len(state.construction) > self.max_construction_len:
                continue
            self._states.append(state)
            existing_ids.add(state.id)
            if key is not None:
                existing_keys.add(key)

        self._states.sort(key=lambda state: float(state.value), reverse=True)
        self._states = self._states[: self.max_buffer_size]

    def _state_by_id(self, state_id: str) -> DiscoveryState:
        for state in self._states:
            if state.id == state_id:
                return state
        raise KeyError(f"Unknown state_id: {state_id}")

    def _set_parent(self, child: DiscoveryState, parent: DiscoveryState) -> None:
        child.parent_values = [float(parent.value)] + list(parent.parent_values)
        child.parents = [{"id": parent.id, "timestep": parent.timestep}] + list(parent.parents)

    def _state_key(self, state: DiscoveryState) -> tuple[Any, ...] | str | None:
        if state.construction:
            return _freeze_jsonable(state.construction)
        if state.code:
            return state.code
        return None

    def _children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for state in self._states:
            for parent in state.parents:
                parent_id = parent.get("id")
                if parent_id:
                    children.setdefault(str(parent_id), set()).add(state.id)
        return children

    def _lineage_ids(self, state: DiscoveryState, children_map: dict[str, set[str]]) -> set[str]:
        lineage = {state.id}
        for parent in state.parents:
            parent_id = parent.get("id")
            if parent_id:
                lineage.add(str(parent_id))
        queue = [state.id]
        visited = {state.id}
        while queue:
            state_id = queue.pop(0)
            for child_id in children_map.get(state_id, set()):
                if child_id not in visited:
                    visited.add(child_id)
                    lineage.add(child_id)
                    queue.append(child_id)
        return lineage

    def _blocked_ids_for_group(self, group_uid: str) -> set[str]:
        step_prefix = group_uid.split(":", 1)[0]
        children_map = self._children_map()
        blocked: set[str] = set()
        for existing_uid, group in self._groups.items():
            if group.get("finalized") or existing_uid == group_uid or not existing_uid.startswith(f"{step_prefix}:"):
                continue
            try:
                state = self._state_by_id(group["state_id"])
            except KeyError:
                continue
            blocked.update(self._lineage_ids(state, children_map))
        return blocked

    def _refresh_best(self) -> None:
        if not self._states:
            self._best_state_id = None
            return
        self._states.sort(key=lambda state: float(state.value), reverse=True)
        self._best_state_id = self._states[0].id

    def _to_store(self) -> dict[str, Any]:
        return {
            "states": [state.to_dict() for state in self._states],
            "groups": self._groups,
            "puct_n": self._puct_n,
            "puct_m": self._puct_m,
            "puct_T": self._puct_T,
            "best_state_id": self._best_state_id,
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "topk_children": self.topk_children,
            "max_buffer_size": self.max_buffer_size,
            "max_construction_len": self.max_construction_len,
            "sample_stats": self.sample_stats(),
            "last_sampled_stats": self._last_sampled_stats,
        }

    def _load(self) -> None:
        store = json.loads(self.path.read_text())
        self._states = [DiscoveryState.from_dict(data) for data in store.get("states", [])]
        self._groups = dict(store.get("groups", {}))
        self._puct_n = {str(k): int(v) for k, v in store.get("puct_n", {}).items()}
        self._puct_m = {str(k): float(v) for k, v in store.get("puct_m", {}).items()}
        self._puct_T = int(store.get("puct_T", 0))
        self._best_state_id = store.get("best_state_id")
        self.max_construction_len = store.get("max_construction_len", self.max_construction_len)
        self._last_sampled_stats = list(store.get("last_sampled_stats", []))

    def _reload_from_disk(self) -> None:
        if self.path.exists():
            self._load()

    @contextmanager
    def _file_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a") as lock_file:
            flock(lock_file.fileno(), LOCK_EX)
            try:
                yield
            finally:
                flock(lock_file.fileno(), LOCK_UN)

    def _save(self, *, snapshot_step: int | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        store = self._to_store()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(store, indent=2, sort_keys=True))
        os.replace(tmp_path, self.path)
        self._write_puct_stats(store)
        if snapshot_step is not None:
            self._write_snapshot(store, snapshot_step)
        if self._best_state_id is not None:
            best_path = self.path.with_name("best_state.json")
            best_state = self._state_by_id(self._best_state_id)
            best_tmp = best_path.with_suffix(".json.tmp")
            best_tmp.write_text(json.dumps(best_state.to_dict(), indent=2, sort_keys=True))
            os.replace(best_tmp, best_path)

    def _write_puct_stats(self, store: dict[str, Any]) -> None:
        stats_path = self.path.with_name("puct_stats.json")
        tmp_path = stats_path.with_suffix(".json.tmp")
        payload = {
            "puct_T": self._puct_T,
            "best_state_id": self._best_state_id,
            "sample_stats": store["sample_stats"],
            "last_sampled_stats": self._last_sampled_stats,
        }
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp_path, stats_path)

    def _write_snapshot(self, store: dict[str, Any], step: int) -> None:
        snapshots_dir = self.path.with_name("archive_snapshots")
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshots_dir / f"step_{step:06d}.json"
        tmp_path = snapshot_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(store, indent=2, sort_keys=True))
        os.replace(tmp_path, snapshot_path)

    def sample_stats(self) -> dict[str, Any]:
        return {
            "puct/buffer_size": len(self._states),
            "puct/T": self._puct_T,
            **_stats([state.value for state in self._states], "puct/buffer_value"),
            **_stats([state.timestep for state in self._states], "puct/buffer_timestep"),
            **_stats([len(state.construction) if state.construction else 0 for state in self._states], "puct/buffer_construction_len"),
        }


def _freeze_jsonable(value: Any) -> tuple[Any, ...] | Any:
    if isinstance(value, list):
        return tuple(_freeze_jsonable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_jsonable(item)) for key, item in value.items()))
    return value


def _stats(values: list[Any], prefix: str) -> dict[str, float]:
    arr = np.array([float(value) for value in values if value is not None], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        f"{prefix}/mean": float(np.mean(arr)),
        f"{prefix}/std": float(np.std(arr)),
        f"{prefix}/min": float(np.min(arr)),
        f"{prefix}/max": float(np.max(arr)),
    }


def _step_from_group_uid(group_uid: str) -> int | None:
    try:
        return int(str(group_uid).split(":", 1)[0])
    except (TypeError, ValueError):
        return None
