from __future__ import annotations
from typing import Any
import numpy as np
from helpers.libdipum.covmatrix import covmatrix


def mahalanobis(*args: Any):
    """Compute Mahalanobis distances.

    Usage
    -----
    D = mahalanobis(Y, X)
        Distances from each row in Y to the centroid/covariance of rows in X.

    D = mahalanobis(Y, Cx, mx)
        Distances from each row in Y using provided covariance matrix Cx and
        mean vector mx.

    Returns
    -------
    D : ndarray, shape (K,)
        Real Mahalanobis distances for each row vector in Y.
    """
    if len(args) not in (2, 3):
        raise ValueError("Wrong number of inputs.")

    Y = np.asarray(args[0])

    if len(args) == 2:
        X = np.asarray(args[1])
        Cx, mx = covmatrix(X)
    else:
        Cx = np.asarray(args[1])
        mx = np.asarray(args[2])

    # Ensure row mean for broadcasting with Y rows.
    mx = np.asarray(mx).reshape(-1)

    if Y.ndim != 2:
        raise ValueError("Y must be a 2-D array of row vectors.")
    if Cx.ndim != 2 or Cx.shape[0] != Cx.shape[1]:
        raise ValueError("Cx must be a square covariance matrix.")
    if Y.shape[1] != Cx.shape[0] or mx.size != Cx.shape[0]:
        raise ValueError("Dimension mismatch among Y, Cx, and mx.")

    # Subtract mean vector from each row of Y.
    Yc = Y - mx

    # MATLAB right division: Yc / Cx == Yc * inv(Cx)
    # Use solve on transpose for numerical stability.
    Yc_over_Cx = np.linalg.solve(Cx.T, Yc.T).T

    # D = real(sum((Yc/Cx).*conj(Yc), 2))
    D = np.real(np.sum(Yc_over_Cx * np.conj(Yc), axis=1))
    return D


__all__ = ["mahalanobis"]
