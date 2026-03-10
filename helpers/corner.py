"""MATLAB-like corner detector.

Implements corner() behavior using cornermetric + local-peak selection.
"""

from __future__ import annotations
from typing import Any

import numpy as np
from scipy import ndimage

try:
    from .cornermetric import cornermetric
except Exception:
    from cornermetric import cornermetric


def _parse_corner_args(I: Any, args: Any, kwargs: Any):
    """_parse_corner_args."""
    method = "Harris"
    max_corners = 200
    quality_level = 0.01
    filter_coef = None
    sensitivity_factor = 0.04
    sensitivity_specified = False

    # Positional compatibility:
    # corner(I)
    # corner(I, METHOD)
    # corner(I, N)
    # corner(I, METHOD, N)
    if len(args) >= 1:
        a0 = args[0]
        if isinstance(a0, str):
            method = a0
        elif np.isscalar(a0):
            method = "Harris"
            max_corners = int(a0)
        else:
            raise ValueError("Second argument must be METHOD (str) or N (int)")

    if len(args) >= 2:
        a1 = args[1]
        if isinstance(args[0], str) and np.isscalar(a1):
            max_corners = int(a1)
        else:
            raise ValueError("Third argument must be N when METHOD is provided")

    if len(args) > 2:
        raise ValueError("Too many positional arguments")

    # Name-value compatibility (case-insensitive keys)
    lower_keys = {str(k).lower(): k for k in kwargs.keys()}

    if "filtercoefficients" in lower_keys:
        filter_coef = kwargs[lower_keys["filtercoefficients"]]
    if "qualitylevel" in lower_keys:
        quality_level = float(kwargs[lower_keys["qualitylevel"]])
    if "sensitivityfactor" in lower_keys:
        sensitivity_factor = float(kwargs[lower_keys["sensitivityfactor"]])
        sensitivity_specified = True
    if "method" in lower_keys:
        method = kwargs[lower_keys["method"]]
    if "n" in lower_keys:
        max_corners = int(kwargs[lower_keys["n"]])

    method = str(method)

    if max_corners <= 0:
        raise ValueError("N must be a positive integer")
    if not (0.0 < quality_level < 1.0):
        raise ValueError("QualityLevel must satisfy 0 < Q < 1")

    if str(method).lower() != "harris" and sensitivity_specified:
        raise ValueError("SensitivityFactor is only valid with Harris method")

    return I, method, sensitivity_factor, filter_coef, max_corners, quality_level


def _suppress_low_corner_metric_maxima(cmetric: Any, bw: Any, quality_level: Any):
    """_suppress_low_corner_metric_maxima."""
    max_cmetric = float(np.max(cmetric)) if cmetric.size else 0.0
    if max_cmetric > 0:
        min_metric = quality_level * max_cmetric
        bw = bw & (cmetric >= min_metric)
    else:
        bw[:] = False
    return bw


def _thin_plateaus_to_single_pixel(bw: Any, cmetric: Any):
    """MATLAB bwmorph(...,'shrink',Inf)-like simplification for local maxima.

    For each connected local-max component (8-connectivity), keep one pixel
    at the maximum corner metric.
    """
    structure = np.ones((3, 3), dtype=bool)
    labels, nlab = ndimage.label(bw, structure=structure)
    out = np.zeros_like(bw, dtype=bool)

    for lab in range(1, nlab + 1):
        rr, cc = np.where(labels == lab)
        if rr.size == 0:
            continue
        vals = cmetric[rr, cc]
        k = int(np.argmax(vals))
        out[rr[k], cc[k]] = True

    return out


def _find_local_peak(cmetric: Any, quality_level: Any):
    """_find_local_peak."""
    nr, nc = cmetric.shape
    if nr < 4 or nc < 4:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    # Regional maxima with 8-connectivity.
    local_max = ndimage.maximum_filter(cmetric, size=3, mode="nearest")
    bw = np.isfinite(cmetric) & (cmetric == local_max)

    bw = _suppress_low_corner_metric_maxima(cmetric, bw, quality_level)
    bw = _thin_plateaus_to_single_pixel(bw, cmetric)

    ind = np.flatnonzero(bw)
    if ind.size == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    order = np.argsort(cmetric.ravel()[ind])[::-1]
    ind = ind[order]
    r, c = np.unravel_index(ind, bw.shape)

    # MATLAB-compatible coordinates: 1-based [x y] i.e., [col row]
    return c + 1, r + 1


def corner(I: Any, *args: Any, **kwargs: Any):
    """Find corner points in an image (MATLAB-like API).

    Returns
    -------
    corners : (M,2) ndarray
        Columns are [x, y] coordinates (1-based, MATLAB-style).
    """
    _, method, sensitivity_factor, filter_coef, max_corners, quality_level = (
        _parse_corner_args(I, args, kwargs)
    )

    if str(method).lower() == "harris":
        cmetric = cornermetric(
            I,
            method=method,
            sensitivity_factor=sensitivity_factor,
            filter_coefficients=filter_coef,
        )
    else:
        cmetric = cornermetric(I, method=method, filter_coefficients=filter_coef)

    xpeak, ypeak = _find_local_peak(cmetric, quality_level)
    corners = np.column_stack((xpeak, ypeak)).astype(np.float64, copy=False)

    if corners.shape[0] > max_corners:
        corners = corners[:max_corners, :]

    return corners


__all__ = ["corner"]
