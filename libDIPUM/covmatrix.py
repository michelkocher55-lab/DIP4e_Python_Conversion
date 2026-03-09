from typing import Any
import numpy as np


def covmatrix(X: Any):
    """
    Computes the covariance matrix and mean vector.

    Parameters
    ----------
    X : array_like, shape (K, N)
        Vector population organized as rows (K samples, N dimensions).

    Returns
    -------
    C : ndarray, shape (N, N)
        Unbiased covariance estimate.
    m : ndarray, shape (N, 1)
        Mean vector (column).

    Notes
    -----
    If K == 1, C is returned as an N-by-N matrix of NaN values,
    matching the documented MATLAB behavior for the unbiased estimate.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix of row vectors.")

    K = X.shape[0]
    if K < 1:
        raise ValueError("X must contain at least one sample (row).")

    # Unbiased estimate of mean.
    m = np.sum(X, axis=0) / K

    # Subtract the mean from each row.
    Xc = X - m

    # Unbiased covariance estimate.
    if K == 1:
        N = X.shape[1]
        C = np.full((N, N), np.nan, dtype=float)
    else:
        C = (Xc.T @ Xc) / (K - 1)

    # Mean as column vector.
    m = m.reshape(-1, 1)

    return C, m
