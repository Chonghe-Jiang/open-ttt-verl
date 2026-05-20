import math
import heapq
import random
from typing import List, Tuple, Optional

import numpy as np


class SimpleProgress:
    def __init__(self, total: int, every: int = 1000):
        self.total = int(total)
        self.every = max(1, int(every))
        self._last = -1

    def update(self, i: int) -> None:
        if i // self.every != self._last:
            self._last = i // self.every
            pct = 100.0 * (i + 1) / float(self.total)
            print(f"add progress: {i + 1}/{self.total} ({pct:.1f}%)")


class GPT5HNSWPaperFast:
    """
    A faster insertion variant of the paper-style HNSW with:
    - Pre-allocated level arrays to avoid repeated resizing
    - Vectorized distance calculation blocks
    - Periodic progress printing
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 128,
        ef_search: int = 64,
        seed: int = 123,
        progress_every: int = 5000,
    ) -> None:
        self.dim = int(dim)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._rng = random.Random(seed)
        self._level_mult = 1.0 / max(1e-6, math.log(max(2.0, float(self.M))))

        self._data: Optional[np.ndarray] = None
        self._levels: List[List[List[int]]] = []
        self._ep: Optional[int] = None
        self._max_level: int = -1
        self._progress_every = progress_every

    @property
    def ntotal(self) -> int:
        return 0 if self._data is None else int(self._data.shape[0])

    def _gen_level(self) -> int:
        r = random.random()
        return int(-math.log(r) * self._level_mult)

    @staticmethod
    def _l2sqr(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))

    def _ensure_levels_for_n(self, n: int) -> None:
        # Ensure each existing level has adjacency lists for n nodes
        for l in range(len(self._levels)):
            cur = len(self._levels[l])
            if cur < n:
                self._levels[l].extend([] for _ in range(n - cur))

    def _ensure_level_exists(self, level: int, n: int) -> None:
        while level >= len(self._levels):
            self._levels.append([[] for _ in range(n)])

    def _greedy_search_layer(self, q: np.ndarray, ep: int, level: int) -> int:
        cur = ep
        cur_dist = self._l2sqr(q, self._data[cur])
        changed = True
        while changed:
            changed = False
            nbrs = self._levels[level][cur]
            if not nbrs:
                break
            X = self._data[np.array(nbrs, dtype=np.int64)]
            diffs = X - q[None, :]
            dists = np.einsum('ij,ij->i', diffs, diffs)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < cur_dist:
                cur = nbrs[min_idx]
                cur_dist = float(dists[min_idx])
                changed = True
        return cur

    def _search_layer(self, q: np.ndarray, entry_points: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        cand: List[Tuple[float, int]] = []
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
            nbrs = self._levels[level][c]
            if nbrs:
                X = self._data[np.array([j for j in nbrs if j not in visited], dtype=np.int64)]
                # If all visited, X may be empty; guard
                if X.size:
                    base_ids = np.array([j for j in nbrs if j not in visited], dtype=np.int64)
                    for j in base_ids:
                        visited.add(int(j))
                    diffs = X - q[None, :]
                    dists = np.einsum('ij,ij->i', diffs, diffs)
                    for j, dj in zip(base_ids, dists):
                        if len(res) < ef or dj < -res[0][0]:
                            heapq.heappush(cand, (float(dj), int(j)))
                            heapq.heappush(res, (-float(dj), int(j)))
                            if len(res) > ef:
                                heapq.heappop(res)
        out = [(-d, i) for d, i in res]
        out.sort(key=lambda t: t[0])
        return out

    def _select_neighbors_heuristic(self, q: np.ndarray, candidates: List[Tuple[float, int]], M: int) -> List[int]:
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
            for dq, cid in candidates:
                if cid not in selected:
                    selected.append(cid)
                    if len(selected) >= M:
                        break
        return selected

    def _trim_neighbors(self, level: int, node: int) -> None:
        nbrs = self._levels[level][node]
        if len(nbrs) <= self.M:
            return
        X = self._data[np.array(nbrs, dtype=np.int64)]
        base = self._data[node]
        diffs = X - base[None, :]
        dists = np.einsum('ij,ij->i', diffs, diffs)
        order = np.argsort(dists)
        keep = [int(nbrs[i]) for i in order[: self.M]]
        self._levels[level][node] = keep

    def add(self, xb: np.ndarray) -> None:
        assert xb.dtype == np.float32 and xb.ndim == 2 and xb.shape[1] == self.dim
        n_new = xb.shape[0]
        if self._data is None:
            self._data = xb.copy()
            n = self.ntotal
            self._ensure_level_exists(0, n)
            self._ep = 0 if n > 0 else None
            self._max_level = 0 if n > 0 else -1
            # connect initial points greedily at base layer only
            prog = SimpleProgress(n, every=max(1, n // 20) or 1)
            for i in range(n):
                if i == 0:
                    continue
                ep = self._ep
                ep = self._greedy_search_layer(self._data[i], ep, 0)
                top = self._search_layer(self._data[i], [ep], self.ef_construction, 0)
                nbrs = self._select_neighbors_heuristic(self._data[i], top, self.M)
                for nb in nbrs:
                    self._levels[0][i].append(nb)
                    self._levels[0][nb].append(i)
                    if len(self._levels[0][nb]) > self.M:
                        self._trim_neighbors(0, nb)
                if len(self._levels[0][i]) > self.M:
                    self._trim_neighbors(0, i)
                prog.update(i)
            return

        # append data and pre-extend structures
        n_old = self.ntotal
        self._data = np.vstack([self._data, xb])
        n_total = self.ntotal
        self._ensure_levels_for_n(n_total)
        self._ensure_level_exists(0, n_total)
        prog = SimpleProgress(n_new, every=max(1, n_new // 20) or 1)

        for t in range(n_new):
            i = n_old + t
            # simplified: assign all new nodes only to base layer for speed
            # (can be extended to multi-layer by sampling level and connecting upper levels similarly)
            if self._ep is None:
                self._ep = i
                self._max_level = 0
                self._ensure_level_exists(0, self.ntotal)
                prog.update(t)
                continue
            ep = self._ep
            ep = self._greedy_search_layer(self._data[i], ep, 0)
            top = self._search_layer(self._data[i], [ep], self.ef_construction, 0)
            nbrs = self._select_neighbors_heuristic(self._data[i], top, self.M)
            for nb in nbrs:
                self._levels[0][i].append(nb)
                self._levels[0][nb].append(i)
                if len(self._levels[0][nb]) > self.M:
                    self._trim_neighbors(0, nb)
            if len(self._levels[0][i]) > self.M:
                self._trim_neighbors(0, i)
            prog.update(t)

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
            ep = self._greedy_search_layer(q, ep, 0)
            top = self._search_layer(q, [ep], max(1, self.ef_search), 0)
            take = min(k, len(top))
            for t in range(take):
                D[qi, t] = top[t][0]
                I[qi, t] = top[t][1]
        return D, I
