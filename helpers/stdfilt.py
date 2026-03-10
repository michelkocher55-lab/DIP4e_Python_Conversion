from typing import Any
import numpy as np
from scipy.ndimage import convolve


def _validate_nhood(h: Any):
    """_validate_nhood."""
    h = np.asarray(h)
    if h.ndim < 1:
        raise ValueError("NHOOD must be at least 1-D.")

    # Must contain only zeros and/or ones.
    if not np.all((h == 0) | (h == 1)):
        raise ValueError("NHOOD must contain only 0 and 1 values.")

    # Each neighborhood dimension must be odd.
    if np.any((np.array(h.shape) % 2) == 0):
        raise ValueError("NHOOD size must be odd in each dimension.")

    return h.astype(np.float64, copy=False)


def stdfilt(I: Any, nhood: Any = None):
    """
    Local standard deviation of image (simplified MATLAB-like stdfilt).

    Parameters
    ----------
    I : ndarray
        Real numeric/logical input array.
    nhood : ndarray, optional
        0/1 neighborhood mask with odd size in each dimension.
        If omitted, a 3x3 (2-D) or 3x...x3 (N-D) all-ones neighborhood is used.

    Returns
    -------
    J : ndarray
        Local standard deviation image (float64), same shape as I.

    Notes
    -----
    - Uses symmetric-like boundary handling via scipy mode='reflect'.
    - This implementation is intentionally simpler than MATLAB internals.
    """
    I = np.asarray(I)
    if np.iscomplexobj(I):
        raise ValueError("I must be real.")

    I = I.astype(np.float64, copy=False)

    if nhood is None:
        # MATLAB default is ones(3) for 2-D; for N-D we extend naturally.
        if I.ndim <= 2:
            h = np.ones((3, 3), dtype=np.float64)
        else:
            h = np.ones((3,) * I.ndim, dtype=np.float64)
    else:
        h = _validate_nhood(nhood)

    if h.ndim != I.ndim:
        raise ValueError("NHOOD must have the same number of dimensions as I.")

    denom = np.sum(h)
    if denom <= 0:
        raise ValueError("NHOOD must contain at least one nonzero element.")

    # E[X] and E[X^2] over neighborhood, then std = sqrt(E[X^2] - E[X]^2).
    mean = convolve(I, h, mode="reflect") / denom
    mean2 = convolve(I * I, h, mode="reflect") / denom
    var = np.maximum(mean2 - mean * mean, 0.0)

    J = np.sqrt(var)
    return J
