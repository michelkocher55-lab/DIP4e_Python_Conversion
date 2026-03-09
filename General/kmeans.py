from typing import Any
import numpy as np


def kmeans(X: Any, k: Any, max_iters: Any = 100, tol: Any = 1e-4, replicates: Any = 1):
    """
    K-means clustering implementation (Lloyd's algorithm).

    Parameters:
        X: Input data (N x M) or (N,)
        k: Number of clusters

    Returns:
        idx: Cluster indices (1-based), shape (N,)
        C: Centroids (k x M)
    """
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    N, M = X.shape

    # Simple initialization: random choice
    # For robustness, could retry 'replicates' times.

    best_inertia = np.inf
    best_idx = None
    best_C = None

    rng = np.random.default_rng(42)

    for _ in range(replicates):
        # Choose k random points as initial centroids
        initial_indices = rng.choice(N, k, replace=False)
        C = X[initial_indices].astype(float)

        for i in range(max_iters):
            # Assign points to nearest centroid
            # Distances: (N, k). squared euclidean
            # (x - c)^2 = x^2 + c^2 - 2xc
            # Or use broadcasting

            # shape (N, 1, M) - (1, k, M) -> (N, k, M)
            diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
            dists = np.sum(diffs**2, axis=2)

            idx_0 = np.argmin(dists, axis=1)

            # Update centroids
            new_C = np.zeros_like(C)
            for j in range(k):
                mask = idx_0 == j
                if np.any(mask):
                    new_C[j] = X[mask].mean(axis=0)
                else:
                    # Handle empty cluster: re-init? or keep properties?
                    # simple strategy: keep old or random re-init
                    new_C[j] = X[rng.choice(N)]

            # Check convergence
            shift = np.sum((new_C - C) ** 2)
            C = new_C
            if shift < tol:
                break

        # Calculate inertia
        # dists updated with final C? No, need to recompute
        diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.sum(diffs**2, axis=2)
        idx_0 = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        inertia = np.sum(min_dists)

        if inertia < best_inertia:
            best_inertia = inertia
            best_idx = idx_0
            best_C = C

    return best_idx + 1, best_C
