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

from erdos_slime.ttt_discover.state import DiscoveryState


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
    ) -> None:
        self.path = Path(path)
        self.rollout_n = int(rollout_n)
        self.puct_c = float(puct_c)
        self.topk_children = int(topk_children)
        self.max_buffer_size = int(max_buffer_size)
        self._lock = threading.RLock()

        self._states: list[DiscoveryState] = []
        self._groups: dict[str, dict[str, Any]] = {}
        self._puct_n: dict[str, int] = {}
        self._puct_m: dict[str, float] = {}
        self._puct_T = 0
        self._best_state_id: str | None = None

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

                state = self._sample_state()
                self._groups[group_uid] = {
                    "state_id": state.id,
                    "children": [],
                    "submitted": 0,
                    "finalized": False,
                }
                self._save()
                return state

    def submit_child(self, group_uid: str, child: DiscoveryState | None) -> bool:
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
                if finalized:
                    self._finalize_group(group_uid)
                self._save()
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

        ancestor_ids = [parent.id] + [str(parent_data["id"]) for parent_data in parent.parents if parent_data.get("id")]
        for state_id in ancestor_ids:
            self._puct_n[state_id] = self._puct_n.get(state_id, 0) + 1
        self._puct_T += 1
        group["finalized"] = True
        self._refresh_best()

    def _sample_state(self) -> DiscoveryState:
        if not self._states:
            raise ValueError("PUCTArchive has no states to sample")
        values = np.array([float(state.value) for state in self._states], dtype=np.float64)
        scale = max(float(values.max() - values.min()), 1e-6)
        ranks = np.argsort(np.argsort(-values))
        weights = (len(values) - ranks).astype(np.float64)
        priors = weights / weights.sum()
        sqrt_t = math.sqrt(1.0 + self._puct_T)

        scored = []
        for idx, state in enumerate(self._states):
            visits = self._puct_n.get(state.id, 0)
            q_value = self._puct_m.get(state.id, float(state.value)) if visits > 0 else float(state.value)
            bonus = self.puct_c * scale * float(priors[idx]) * sqrt_t / (1.0 + visits)
            scored.append((q_value + bonus, float(state.value), state))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def _add_states(self, states: list[DiscoveryState]) -> None:
        existing = {state.id for state in self._states}
        for state in states:
            if state.id not in existing:
                self._states.append(state)
                existing.add(state.id)

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
        }

    def _load(self) -> None:
        store = json.loads(self.path.read_text())
        self._states = [DiscoveryState.from_dict(data) for data in store.get("states", [])]
        self._groups = dict(store.get("groups", {}))
        self._puct_n = {str(key): int(value) for key, value in store.get("puct_n", {}).items()}
        self._puct_m = {str(key): float(value) for key, value in store.get("puct_m", {}).items()}
        self._puct_T = int(store.get("puct_T", 0))
        self._best_state_id = store.get("best_state_id")

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

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        store = self._to_store()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(store, indent=2, sort_keys=True))
        os.replace(tmp_path, self.path)
        if self._best_state_id is not None:
            best_path = self.path.with_name("best_state.json")
            best_state = self._state_by_id(self._best_state_id)
            best_tmp = best_path.with_suffix(".json.tmp")
            best_tmp.write_text(json.dumps(best_state.to_dict(), indent=2, sort_keys=True))
            os.replace(best_tmp, best_path)
