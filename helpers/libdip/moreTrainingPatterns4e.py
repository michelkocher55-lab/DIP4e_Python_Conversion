"""Increase size of a training set (MATLAB moreTrainingPatterns4e equivalent)."""

from __future__ import annotations

import numpy as np

from helpers.libdip.patternShuffle4e import patternShuffle4e


def moreTrainingPatterns4e(X, R, nt):
    """Increase training patterns by repetition and shuffle.

    This follows the provided MATLAB implementation exactly:
    `newX = repmat(X,1,nt)`, `newR = repmat(R,1,nt)`, then shuffle.

    Parameters
    ----------
    X : array_like, shape (n, np)
        Pattern vectors as columns.
    R : array_like, shape (c, np)
        Response/class vectors as columns.
    nt : float or int
        Replication factor (rounded to nearest integer).

    Returns
    -------
    newX : ndarray
        Replicated and shuffled pattern matrix.
    newR : ndarray
        Replicated and shuffled response matrix.
    """
    X = np.asarray(X)
    R = np.asarray(R)

    if X.ndim != 2 or R.ndim != 2:
        raise ValueError("X and R must be 2-D matrices.")
    if X.shape[1] != R.shape[1]:
        raise ValueError("X and R must have the same number of columns.")

    nt = int(np.round(nt))
    if nt < 0:
        raise ValueError("nt must be nonnegative.")

    newX = np.tile(X, (1, nt))
    newR = np.tile(R, (1, nt))

    newX, newR, _ = patternShuffle4e(newX, newR)
    return newX, newR


__all__ = ["moreTrainingPatterns4e"]
