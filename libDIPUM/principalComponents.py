from typing import Any
import numpy as np

from libDIPUM.covmatrix import covmatrix


def principalComponents(X: Any, q: Any):
    """
    Principal components of a vector population.

    Parameters
    ----------
    X : array_like, shape (K, n)
        Row-wise population vectors.
    q : int
        Number of principal components to keep (0 <= q <= n).

    Returns
    -------
    P : dict
        Structure-like dictionary with fields:
        Y, A, X, ems, Cx, mx, Cy, d, V.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix of row vectors.")

    K, n = X.shape

    if int(q) != q:
        raise ValueError("q must be an integer.")
    q = int(q)
    if q < 0 or q > n:
        raise ValueError("q must be in the range [0, n].")

    P = {}

    # Mean vector and covariance matrix.
    Cx, mx_col = covmatrix(X)
    P["Cx"] = Cx

    # MATLAB converts mean to row, uses it in computations, then back to col.
    mx_row = mx_col.reshape(1, -1)

    # Eigen-decomposition; Cx is symmetric, so use eigh.
    d, V = np.linalg.eigh(P["Cx"])

    # Sort eigenvalues in descending order and reorder eigenvectors.
    idx = np.argsort(d)[::-1]
    d = d[idx]
    V = V[:, idx]

    # Transformation matrix A: q rows from first q eigenvectors.
    A = V[:, :q].T if q > 0 else np.zeros((0, n), dtype=float)
    P["A"] = A

    # Principal component vectors.
    # MATLAB: Y = A*(X - mx_row)'; then Y = Y'
    Y_q_by_K = A @ (X - mx_row).T

    # Reconstructed vectors.
    Xrec = (A.T @ Y_q_by_K).T + mx_row

    # Convert Y to K-by-q and mean back to n-by-1.
    P["Y"] = Y_q_by_K.T
    P["X"] = Xrec
    P["mx"] = mx_row.reshape(-1, 1)

    # Mean square error: sum of discarded eigenvalues.
    P["ems"] = float(np.sum(d[q:]))

    # Covariance matrix of Y.
    P["Cy"] = A @ P["Cx"] @ A.T

    # Extra fields kept from MATLAB source comments.
    P["d"] = d
    P["V"] = V

    return P
