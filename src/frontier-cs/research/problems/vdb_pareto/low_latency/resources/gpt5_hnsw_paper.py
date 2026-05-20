import math
import heapq
import random
from typing import List, Tuple, Optional

import numpy as np


class GPT5HNSWPaper:
    """
    A simplified HNSW implementation following the core ideas in
    "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
    by Malkov & Yashunin (2016).

    - Multi-layer graph with exponentially distributed levels per element
    - Greedy descent on upper layers, efConstruction search + neighbor selection on insertion
    - efSearch search on base layer for queries

    Design notes:
    - This is a didactic CPU/Python implementation meant for benchmarking integration.
      It is not optimized; large datasets will be slow to build.
    - Only L2 distance is supported.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 123,
    ) -> None:
        self.dim = int(dim)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._rng = random.Random(seed)

        # level generation multiplier as in common HNSW implementations
        # level ~ floor(-ln(U) * mL), mL ≈ 1 / ln(M)
        self._level_mult = 1.0 / max(1e-6, math.log(max(2.0, float(self.M))))

        self._data: Optional[np.ndarray] = None  # shape (n, dim), float32
        # adjacency per level: levels[l][i] -> List[int]
        self._levels: List[List[List[int]]] = []
        self._ep: Optional[int] = None
        self._max_level: int = -1

    @property
    def ntotal(self) -> int:
        return 0 if self._data is None else int(self._data.shape[0])

    # ---------------------------- Utilities ----------------------------
    def _gen_level(self) -> int:
        r = random.random()
        return int(-math.log(r) * self._level_mult)

    @staticmethod
    def _l2sqr(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))

    def _ensure_levels_capacity_for_new_node(self, new_count: int, node_level: int) -> None:
        # Ensure there are node slots for each existing level
        for l in range(len(self._levels)):
            self._levels[l].extend([] for _ in range(new_count))
        # Add new levels if node_level exceeds current
        while node_level >= len(self._levels):
            # create a new level with empty adjacency for all nodes including the upcoming one(s)
            n_after_append = self.ntotal + new_count
            self._levels.append([[] for _ in range(n_after_append)])

    # ------------------------ Core search routines ---------------------
    def _greedy_search_layer(self, q: np.ndarray, ep: int, level: int) -> int:
        cur = ep
        cur_dist = self._l2sqr(q, self._data[cur])
        changed = True
        while changed:
            changed = False
            for j in self._levels[level][cur]:
                d = self._l2sqr(q, self._data[j])
                if d < cur_dist:
                    cur = j
                    cur_dist = d
                    changed = True
        return cur

    def _search_layer(self, q: np.ndarray, entry_points: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        # Candidate min-heap by distance
        cand: List[Tuple[float, int]] = []
        # Result max-heap by negative distance (so heap[0] is farthest among results)
        res: List[Tuple[float, int]] = []
        visited = set()

        for ep in entry_points:
            d = self._l2sqr(q, self._data[ep])
            heapq.heappush(cand, (d, ep))
            heapq.heappush(res, (-d, ep))
            visited.add(ep)

        while cand:
            d, c = heapq.heappop(cand)
            worst = -res[0][0]
            if d > worst and len(res) >= ef:
                break
            for j in self._levels[level][c]:
                if j in visited:
                    continue
                visited.add(j)
                dj = self._l2sqr(q, self._data[j])
                if len(res) < ef or dj < -res[0][0]:
                    heapq.heappush(cand, (dj, j))
                    heapq.heappush(res, (-dj, j))
                    if len(res) > ef:
                        heapq.heappop(res)

        # convert to ascending list of (dist, id)
        out = [(-d, i) for d, i in res]
        out.sort(key=lambda t: t[0])
        return out

    def _select_neighbors_heuristic(self, q: np.ndarray, candidates: List[Tuple[float, int]], M: int, level: int) -> List[int]:
        # candidates are (dist_to_q, id) in ascending order
        selected: List[int] = []
        for dq, cid in candidates:
            good = True
            cq = dq
            for sid in selected:
                ds = self._l2sqr(self._data[cid], self._data[sid])
                if ds < cq:
                    good = False
                    break
            if good:
                selected.append(cid)
            if len(selected) >= M:
                break
        if len(selected) < M:
            # fill with closest by distance to q (already ordered), skipping duplicates
            for dq, cid in candidates:
                if cid not in selected:
                    selected.append(cid)
                    if len(selected) >= M:
                        break
        return selected

    def _connect_new_node(self, q: np.ndarray, q_id: int, level_q: int) -> None:
        if self._ep is None:
            self._ep = q_id
            self._max_level = level_q
            return

        ep = self._ep
        # Greedy search from top down to level_q+1
        for l in range(self._max_level, level_q, -1):
            ep = self._greedy_search_layer(q, ep, l)

        # For each level down to 0, run efConstruction search and connect
        for l in range(min(self._max_level, level_q), -1, -1):
            top_candidates = self._search_layer(q, [ep], self.ef_construction, l)
            neighbors = self._select_neighbors_heuristic(q, top_candidates, self.M, l)

            # add reciprocal links
            for nb in neighbors:
                self._levels[l][q_id].append(nb)
                self._levels[l][nb].append(q_id)
                # trim degrees to M with simple farthest removal relative to node
                if len(self._levels[l][nb]) > self.M:
                    self._trim_neighbors(l, nb)
            if len(self._levels[l][q_id]) > self.M:
                self._trim_neighbors(l, q_id)

            # update entry point for the next lower layer as the closest among neighbors
            if neighbors:
                ep = min(neighbors, key=lambda nid: self._l2sqr(q, self._data[nid]))

        # possibly update entry point and max level
        if level_q > self._max_level:
            self._ep = q_id
            self._max_level = level_q

    def _trim_neighbors(self, level: int, node: int) -> None:
        # keep up to M neighbors that are closest to "node"
        nbrs = self._levels[level][node]
        if len(nbrs) <= self.M:
            return
        dists = [(self._l2sqr(self._data[node], self._data[j]), j) for j in nbrs]
        dists.sort(key=lambda t: t[0])
        keep = [j for _, j in dists[: self.M]]
        self._levels[level][node] = keep

    # ----------------------------- API --------------------------------
    def add(self, xb: np.ndarray) -> None:
        assert xb.dtype == np.float32 and xb.ndim == 2 and xb.shape[1] == self.dim
        if self._data is None:
            self._data = xb.copy()
            # initialize level containers and connect sequentially
            for i in range(self.ntotal):
                li = self._gen_level()
                self._ensure_levels_capacity_for_new_node(0, li)
                self._connect_new_node(self._data[i], i, li)
            return

        n_old = self.ntotal
        self._data = np.vstack([self._data, xb])
        n_new = xb.shape[0]
        # Provision adjacency lists for new nodes across existing/new levels
        # We will assign levels individually per point and connect below
        # First, ensure each existing level has slots for new nodes
        for l in range(len(self._levels)):
            self._levels[l].extend([] for _ in range(n_new))

        # For each new point, draw level and ensure new level arrays exist
        for t in range(n_new):
            i = n_old + t
            li = self._gen_level()
            self._ensure_levels_capacity_for_new_node(0, li)
            self._connect_new_node(self._data[i], i, li)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert xq.dtype == np.float32 and xq.ndim == 2 and xq.shape[1] == self.dim
        n = self.ntotal
        nq = xq.shape[0]
        D = np.full((nq, k), np.inf, dtype=np.float32)
        I = np.full((nq, k), -1, dtype=np.int64)
        if n == 0 or self._ep is None:
            return D, I

        for qi in range(nq):
            q = xq[qi]
            ep = self._ep
            # Greedy descent from top layer to base layer
            for l in range(self._max_level, 0, -1):
                ep = self._greedy_search_layer(q, ep, l)
            # efSearch exploration on base layer
            top = self._search_layer(q, [ep], max(1, self.ef_search), 0)
            top.sort(key=lambda t: t[0])
            take = min(k, len(top))
            for t in range(take):
                D[qi, t] = top[t][0]
                I[qi, t] = top[t][1]
        return D, I


