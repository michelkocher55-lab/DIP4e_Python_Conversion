"""MATLAB-like cornermetric implementation.

Supports methods:
- Harris
- MinimumEigenvalue
"""

from __future__ import annotations
from typing import Any

import numpy as np
from scipy import ndimage


def _default_filter_coefficients() -> np.ndarray:
    """_default_filter_coefficients."""
    # MATLAB default: fspecial('gaussian',[5 1],1.5)
    sigma = 1.5
    x = np.arange(-2, 3, dtype=np.float64)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g /= g.sum()
    return g


def _convert_to_double(im: np.ndarray) -> np.ndarray:
    """_convert_to_double."""
    a = np.asarray(im)

    if a.dtype == np.uint32:
        return a.astype(np.float64) / np.float64(np.iinfo(np.uint32).max)
    if a.dtype == np.int32:
        return (a.astype(np.float64) + 2.0**31) / (2.0**32)
    if a.dtype == np.int8:
        return (a.astype(np.float64) + 2.0**7) / (2.0**8)

    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        if info.min < 0:
            return (a.astype(np.float64) - info.min) / (info.max - info.min)
        return a.astype(np.float64) / info.max

    if a.dtype == np.bool_:
        return a.astype(np.float64)

    return a.astype(np.float64)


def cornermetric(
    I: Any,
    method: str = "Harris",
    sensitivity_factor: float = 0.04,
    filter_coefficients: Any = None,
):
    """Compute corner metric matrix from an image.

    Parameters
    ----------
    I : array_like
        2-D grayscale/logical image.
    method : {'Harris', 'MinimumEigenvalue'}
    sensitivity_factor : float
        Used only for Harris method. Must satisfy 0 < k < 0.25.
    filter_coefficients : 1-D array_like, optional
        Smoothing vector V. Kernel is V*V'.
    """
    arr = np.asarray(I)
    if arr.ndim != 2:
        raise ValueError("I must be a 2-D array")
    if not np.isrealobj(arr):
        raise ValueError("I must be real")

    m = str(method).strip().lower()
    if m not in ("harris", "minimumeigenvalue"):
        raise ValueError("method must be 'Harris' or 'MinimumEigenvalue'")

    if filter_coefficients is None:
        fcoef = _default_filter_coefficients()
    else:
        fcoef = np.asarray(filter_coefficients, dtype=np.float64).ravel()

    if fcoef.size < 3 or (fcoef.size % 2 == 0):
        raise ValueError("FilterCoefficients length must be odd and >= 3")

    if not (0.0 < float(sensitivity_factor) < 0.25):
        raise ValueError("SensitivityFactor must satisfy 0 < k < 0.25")

    I2 = _convert_to_double(arr)

    # MATLAB: imfilter(..., 'replicate', 'same', 'conv') with [-1 0 1] and transpose.
    gx = ndimage.convolve(
        I2, np.array([[-1.0, 0.0, 1.0]], dtype=np.float64), mode="nearest"
    )
    gy = ndimage.convolve(
        I2, np.array([[-1.0], [0.0], [1.0]], dtype=np.float64), mode="nearest"
    )

    # Keep only valid interior gradients (as in MATLAB code).
    gx = gx[1:-1, 1:-1]
    gy = gy[1:-1, 1:-1]

    A = gx * gx
    B = gy * gy
    C = gx * gy

    # Separable smoothing kernel w = v*v'.
    w = np.outer(fcoef, fcoef)
    A = ndimage.convolve(A, w, mode="nearest")
    B = ndimage.convolve(B, w, mode="nearest")
    C = ndimage.convolve(C, w, mode="nearest")

    # Restore to original image size (MATLAB implementation returns same size as I).
    A = np.pad(A, ((1, 1), (1, 1)), mode="edge")
    B = np.pad(B, ((1, 1), (1, 1)), mode="edge")
    C = np.pad(C, ((1, 1), (1, 1)), mode="edge")

    if m == "harris":
        k = float(sensitivity_factor)
        cornerness = (A * B) - (C * C) - k * (A + B) ** 2
    else:
        cornerness = ((A + B) - np.sqrt((A - B) ** 2 + 4.0 * C * C)) / 2.0

    return cornerness


__all__ = ["cornermetric"]
