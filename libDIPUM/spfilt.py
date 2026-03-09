from typing import Any
import numpy as np
from scipy.ndimage import (
    convolve,
    rank_filter,
    median_filter,
    maximum_filter,
    minimum_filter,
)


# ------------------------------------------------------------
# Utility: tofloat / revertClass (MATLAB-like behavior)
# ------------------------------------------------------------
def tofloat(g: Any):
    """tofloat."""
    orig_dtype = g.dtype
    g = g.astype(np.float64)
    return g, lambda x: x.astype(orig_dtype)


# ------------------------------------------------------------
# Mean filters
# ------------------------------------------------------------
def gmean(g: Any, m: Any, n: Any):
    """gmean."""
    g, revert = tofloat(g)
    kernel = np.ones((m, n))
    f = np.exp(convolve(np.log(g + np.finfo(float).eps), kernel, mode="nearest")) ** (
        1 / (m * n)
    )
    return revert(f)


def harmean(g: Any, m: Any, n: Any):
    """harmean."""
    g, revert = tofloat(g)
    kernel = np.ones((m, n))
    f = (m * n) / convolve(1 / (g + np.finfo(float).eps), kernel, mode="nearest")
    return revert(f)


def charmean(g: Any, m: Any, n: Any, q: Any):
    """charmean."""
    g, revert = tofloat(g)
    kernel = np.ones((m, n))
    # Avoid divide-by-zero/invalid for negative q when g contains zeros (e.g., pepper noise).
    g_safe = np.maximum(g, np.finfo(float).eps)
    num = convolve(g_safe ** (q + 1), kernel, mode="nearest")
    den = convolve(g_safe**q, kernel, mode="nearest") + np.finfo(float).eps
    return revert(num / den)


# ------------------------------------------------------------
# Alpha-trimmed mean filter
# ------------------------------------------------------------
def alphatrim(g: Any, m: Any, n: Any, d: Any):
    """alphatrim."""
    if d <= 0 or d % 2 != 0:
        raise ValueError("d must be a positive even integer")

    g, revert = tofloat(g)
    kernel = np.ones((m, n))

    f = convolve(g, kernel, mode="reflect")

    for k in range(1, d // 2 + 1):
        f -= rank_filter(g, rank=k - 1, size=(m, n), mode="reflect")

    for k in range(m * n - d // 2, m * n):
        f -= rank_filter(g, rank=k, size=(m, n), mode="reflect")

    f /= m * n - d
    return revert(f)


# ------------------------------------------------------------
# Main SPFILT function
# ------------------------------------------------------------
def spfilt(g: Any, type: Any, *args: Any):
    """spfilt."""
    # Defaults (same as MATLAB)
    m = n = 3
    Q = 1.5
    d = 2

    if len(args) > 0:
        m = args[0]
    if len(args) > 1:
        n = args[1]
    if len(args) > 2:
        Q = args[2]
        d = args[2]

    if type == "amean":
        kernel = np.ones((m, n)) / (m * n)
        f = convolve(g, kernel, mode="nearest")

    elif type == "gmean":
        f = gmean(g, m, n)

    elif type == "hmean":
        f = harmean(g, m, n)

    elif type == "chmean":
        f = charmean(g, m, n, Q)

    elif type == "median":
        f = median_filter(g, size=(m, n), mode="reflect")

    elif type == "max":
        f = maximum_filter(g, size=(m, n))

    elif type == "min":
        f = minimum_filter(g, size=(m, n))

    elif type == "midpoint":
        f1 = rank_filter(g, rank=0, size=(m, n), mode="reflect")
        f2 = rank_filter(g, rank=m * n - 1, size=(m, n), mode="reflect")
        f = 0.5 * f1 + 0.5 * f2

    elif type == "atrimmed":
        f = alphatrim(g, m, n, d)

    else:
        raise ValueError("Unknown filter type")

    return f.astype(g.dtype)
