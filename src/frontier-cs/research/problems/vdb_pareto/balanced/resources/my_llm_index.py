import numpy as np

class LLMIndex:
    """
    Simple brute-force ANN implementation.
    Slow but correct - serves as a baseline.
    """
    
    def __init__(self, dim: int, ef_search: int = 64):
        self.dim = int(dim)
        self.ef_search = int(ef_search)
        self._data = None  # shape (n, dim), float32
    
    def add(self, xb: np.ndarray) -> None:
        """Add vectors to the index."""
        assert xb.dtype == np.float32
        assert xb.ndim == 2 and xb.shape[1] == self.dim
        
        if self._data is None:
            self._data = xb.copy()
        else:
            self._data = np.vstack([self._data, xb])
    
    def search(self, xq: np.ndarray, k: int):
        """
        Search for k nearest neighbors using brute force.
        
        Args:
            xq: Query vectors, shape (nq, dim)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices)
        """
        assert xq.dtype == np.float32
        assert xq.ndim == 2 and xq.shape[1] == self.dim
        assert k > 0
        
        nq = xq.shape[0]
        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)
        
        for i in range(nq):
            # Compute L2 distances to all vectors
            diffs = self._data - xq[i][None, :]
            dists = np.einsum('ij,ij->i', diffs, diffs)
            
            # Get k smallest distances
            order = np.argsort(dists)[:k]
            D[i] = dists[order]
            I[i] = order
        
        return D, I
