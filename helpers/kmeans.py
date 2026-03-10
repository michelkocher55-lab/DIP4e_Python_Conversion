from typing import Any
import numpy as np


def kmeans(X: Any, k: Any, max_iters: Any = 100, tol: Any = 1e-4, replicates: Any = 1):
    """
    K-means clustering.

    Parameters:
        X: Input data (1D array or N x M matrix).
        k: Number of clusters.
        max_iters: Maximum iterations.
        tol: Tolerance for convergence.
        replicates: Number of times to repeat simple k-means with new random starts.

    Returns:
        idx: Cluster indices (1-based, matching MATLAB 1..k).
        C: Cluster centers.
    """
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    N, M = X.shape
    best_inertia = np.inf
    best_idx = None
    best_C = None

    rng = np.random.default_rng(42)

    for _ in range(max(1, replicates)):
        # Initialize centers
        # Randomly choose k unique rows if possible
        if N >= k:
            initial_indices = rng.choice(N, k, replace=False)
            C = X[initial_indices].astype(float)
        else:
            C = X[rng.choice(N, k, replace=True)].astype(float)

        for i in range(max_iters):
            # Distances: (N, 1, M) - (1, k, M) -> (N, k, M)
            diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
            # Sum squared diffs
            dists = np.sum(diffs**2, axis=2)  # (N, k)

            # Assign to nearest center
            idx_0 = np.argmin(dists, axis=1)  # 0..k-1

            new_C = np.zeros_like(C)
            converged = True

            for j in range(k):
                mask = idx_0 == j
                if np.any(mask):
                    new_C[j] = X[mask].mean(axis=0)
                else:
                    # Empty cluster, re-init to random point
                    new_C[j] = X[rng.choice(N)]
                    converged = (
                        False  # Force another iter if we had to move a center randomly
                    )

            if converged and np.sum((new_C - C) ** 2) < tol:
                C = new_C
                break
            C = new_C

        # Final inertia
        diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.sum(diffs**2, axis=2)
        idx_0 = np.argmin(dists, axis=1)
        inertia = np.sum(np.min(dists, axis=1))

        if inertia < best_inertia:
            best_inertia = inertia
            best_idx = idx_0
            best_C = C

    if best_idx is None:
        # Fallback if something went wrong
        best_idx = np.zeros(N, dtype=int)
        best_C = np.zeros((k, M))

    return best_idx + 1, best_C
