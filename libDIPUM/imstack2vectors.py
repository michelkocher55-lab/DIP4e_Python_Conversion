from typing import Any
import numpy as np


def imstack2vectors(S: Any, MASK: Any = None):
    """
    Extract vectors from an image stack.

    Parameters
    ----------
    S : ndarray, shape (M, N, n)
        Stack of n registered images.
    MASK : ndarray, optional, shape (M, N)
        Logical/numeric mask. Nonzero elements indicate retained locations.
        If omitted, all M*N locations are used.

    Returns
    -------
    X : ndarray, shape (K, n)
        Extracted vectors as rows.
    r : ndarray, shape (K, 1)
        1-based linear indices (MATLAB-style, column-major) of retained
        locations in the M-by-N plane.
    """
    S = np.asarray(S)
    if S.ndim != 3:
        raise ValueError("S must be an M-by-N-by-n stack array.")

    M, N, n = S.shape

    if MASK is None:
        mask = np.ones((M, N), dtype=bool)
    else:
        mask = np.asarray(MASK) != 0
        if mask.shape != (M, N):
            raise ValueError(
                "MASK must have shape (M, N) matching the first two dimensions of S."
            )

    Q = M * N

    # MATLAB order for reshape/find is column-major ('F').
    X = np.reshape(S, (Q, n), order="F")
    mask_vec = np.reshape(mask, (Q,), order="F")

    X = X[mask_vec, :]

    # MATLAB-style find indices are 1-based and column-major.
    r = np.flatnonzero(mask_vec) + 1
    r = r.reshape(-1, 1)

    return X, r
