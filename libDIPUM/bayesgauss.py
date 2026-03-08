"""Bayes classifier for Gaussian patterns (MATLAB bayesgauss equivalent)."""

from __future__ import annotations

import numpy as np

from libDIPUM.mahalanobis import mahalanobis


def bayesgauss(X, C, M, P=None):
    """Classify row patterns using Bayes Gaussian decision functions.

    Parameters
    ----------
    X : array_like, shape (K, n) or wider
        Input patterns as rows. If wider than n, only first n columns are used,
        where n is inferred from C.
    C : array_like, shape (n, n, Nc)
        Covariance matrices, one per class.
    M : array_like, shape (Nc, n)
        Mean vectors, one per class (rows).
    P : array_like, shape (Nc,), optional
        Class priors. If omitted, classes are assumed equally likely.

    Returns
    -------
    d : ndarray, shape (K,)
        Assigned class number for each input pattern, using MATLAB-compatible
        1-based indexing.
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    M = np.asarray(M, dtype=float)

    if C.ndim != 3:
        raise ValueError("C must be a 3-D array of covariance matrices (n, n, Nc).")

    n = C.shape[0]
    if C.shape[1] != n:
        raise ValueError("Each covariance matrix must be square (n x n).")

    # Protect against possible class labels in extra columns.
    if X.ndim != 2 or X.shape[1] < n:
        raise ValueError("X must be a 2-D array with at least n columns.")
    X = X[:, :n]

    Nc = C.shape[2]
    K = X.shape[0]

    if M.shape != (Nc, n):
        raise ValueError("M must have shape (Nc, n), matching C.")

    if P is None:
        P = np.full(Nc, 1.0 / Nc, dtype=float)
    else:
        P = np.asarray(P, dtype=float).reshape(-1)
        if P.size != Nc:
            raise ValueError("P must contain Nc prior probabilities.")
        if not np.isclose(np.sum(P), 1.0):
            raise ValueError("Elements of P must sum to 1.")

    # Determinants of covariance matrices.
    DM = np.empty(Nc, dtype=float)
    for j in range(Nc):
        DM[j] = np.linalg.det(C[:, :, j])

    # Decision functions.
    D = np.empty((K, Nc), dtype=float)
    for j in range(Nc):
        if P[j] == 0:
            D[:, j] = -np.inf
            continue

        Cm = C[:, :, j]
        Mm = M[j, :]

        L = np.log(P[j])
        DET = 0.5 * np.log(DM[j])
        D[:, j] = L - DET - 0.5 * mahalanobis(X, Cm, Mm)

    # MATLAB chooses class by maximum decision function;
    # ties map to first class index, matching np.argmax behavior.
    d = np.argmax(D, axis=1) + 1  # 1-based labels
    return d


__all__ = ["bayesgauss"]
