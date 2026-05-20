import numpy as np
import random
import heapq


class GPT5HNSW:
    """
    A very simple, toy HNSW-like ANN index implemented in pure Python/NumPy.
    - Graph: single layer, undirected, maximum M edges per node.
    - Construction: each new node connects to up to M random existing nodes (reciprocal links).
      This makes build O(N * M) without heavy distance computations.
    - Search: greedy best-first with a small beam (ef_search), using a min-heap.
    Notes:
      - This implementation is for demonstration and benchmarking integration only.
      - For large datasets (e.g., 1M), consider limiting the number of added points to avoid long Python loops.
    """

    def __init__(self, dim: int, M: int = 16, ef_search: int = 64, seed: int = 123, metric: str = 'L2'):
        assert metric == 'L2', "Only L2 is supported in this toy implementation"
        self.dim = dim
        self.M = int(M)
        self.ef_search = int(ef_search)
        self.metric = metric
        self._rng = random.Random(seed)
        self._data = None            # np.ndarray shape (n, dim), float32
        self._edges = []             # List[List[int]] adjacency

    @property
    def ntotal(self) -> int:
        return 0 if self._data is None else int(self._data.shape[0])

    def add(self, xb: np.ndarray) -> None:
        assert xb.dtype == np.float32
        assert xb.ndim == 2 and xb.shape[1] == self.dim
        n_new = xb.shape[0]
        if self._data is None:
            self._data = xb.copy()
            self._edges = [[] for _ in range(n_new)]
            # connect first nodes among themselves sparsely
            for i in range(n_new):
                self._connect_new_node(i)
            return

        n_old = self._data.shape[0]
        # append data
        self._data = np.vstack([self._data, xb])
        # extend edges list
        self._edges.extend([] for _ in range(n_new))
        # connect each new node
        for local_idx in range(n_new):
            i = n_old + local_idx
            self._connect_new_node(i)

    def _connect_new_node(self, i: int) -> None:
        # choose up to M distinct existing nodes uniformly at random
        if i == 0:
            return
        pool = range(0, i)
        degree = min(self.M, i)
        neighbors = self._rng.sample(pool, degree)
        # add reciprocal links, cap degree to M by random trimming
        for j in neighbors:
            if len(self._edges[i]) < self.M:
                self._edges[i].append(j)
            else:
                # randomly replace one existing edge to keep degree <= M
                idx = self._rng.randrange(self.M)
                self._edges[i][idx] = j
            if len(self._edges[j]) < self.M:
                self._edges[j].append(i)
            else:
                idx = self._rng.randrange(self.M)
                self._edges[j][idx] = i

    def _l2sqr(self, a: np.ndarray, b: np.ndarray) -> float:
        # both shape (dim,)
        diff = a - b
        return float(np.dot(diff, diff))

    def _search_one(self, q: np.ndarray, k: int) -> tuple:
        n = self.ntotal
        if n == 0:
            return np.full(k, np.inf, dtype=np.float32), np.full(k, -1, dtype=np.int64)
        entry = 0
        visited = set([entry])
        d_entry = self._l2sqr(q, self._data[entry])
        # min-heap of (distance, node_id)
        cand = [(d_entry, entry)]
        best = []  # popped order (distance, node_id)
        expansions = 0
        ef = max(1, self.ef_search)
        while cand and expansions < ef:
            d, i = heapq.heappop(cand)
            best.append((d, i))
            nbrs = self._edges[i]
            # expand neighbors not yet visited
            new_nbrs = [j for j in nbrs if j not in visited]
            if new_nbrs:
                # vectorized distance compute for this batch
                X = self._data[np.array(new_nbrs, dtype=np.int64)]  # (m, dim)
                diffs = X - q[None, :]
                dists = np.einsum('ij,ij->i', diffs, diffs)
                for j, dj in zip(new_nbrs, dists):
                    visited.add(j)
                    heapq.heappush(cand, (float(dj), int(j)))
            expansions += 1
        # select k nearest from visited set we popped (best)
        best.sort(key=lambda t: t[0])
        take = min(k, len(best))
        D = np.full(k, np.inf, dtype=np.float32)
        I = np.full(k, -1, dtype=np.int64)
        for t in range(take):
            D[t] = best[t][0]
            I[t] = best[t][1]
        return D, I

    def search(self, xq: np.ndarray, k: int) -> tuple:
        assert xq.dtype == np.float32
        assert xq.ndim == 2 and xq.shape[1] == self.dim
        nq = xq.shape[0]
        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)
        for i in range(nq):
            d, idx = self._search_one(xq[i], k)
            D[i] = d
            I[i] = idx
        return D, I


