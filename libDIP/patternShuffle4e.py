"""Shuffle pattern vectors (MATLAB patternShuffle4e equivalent)."""

from __future__ import annotations
from typing import Any

import numpy as np


def patternShuffle4e(X: Any, R: Any, mode: Any = "random"):
    """Shuffle columns of pattern and response matrices in the same order.

    Parameters
    ----------
    X : array_like, shape (n, np)
        Pattern vectors as columns.
    R : array_like, shape (c, np)
        Class-response vectors as columns.
    mode : {'random', 'repeat'}, optional
        If 'repeat', uses a fixed seed so each call is reproducible.
        If 'random' (default), shuffling differs across calls.

    Returns
    -------
    Xrancol : ndarray
        Shuffled X.
    Rrancol : ndarray
        Shuffled R with the same column permutation.
    order : ndarray
        Zero-based permutation indices used for shuffling.
    """
    X = np.asarray(X)
    R = np.asarray(R)

    if X.ndim != 2 or R.ndim != 2:
        raise ValueError("X and R must be 2-D matrices.")
    if X.shape[1] != R.shape[1]:
        raise ValueError("X and R must have the same number of columns.")

    if mode == "repeat":
        rng = np.random.default_rng(1)
    elif mode == "random" or mode is None:
        rng = np.random.default_rng()
    else:
        raise ValueError("Unknown mode")

    npats = X.shape[1]
    order = rng.permutation(npats)
    Xrancol = X[:, order]
    Rrancol = R[:, order]
    return Xrancol, Rrancol, order


__all__ = ["patternShuffle4e"]
