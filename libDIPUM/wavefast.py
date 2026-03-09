from typing import Any
import numpy as np
from scipy.signal import convolve2d

from libDIPUM.wavefilter import wavefilter
from General.padarray import padarray


def wavefast(x: Any, n: Any, *args: Any):
    """
    Compute the N-level FWT of a 2-D image or 3-D stack of 2-D images.
    MATLAB-faithful translation of DIPUM wavefast.m.
    """
    if len(args) < 1:
        raise ValueError("Missing wavelet/filter arguments.")

    extmode = "SYM"
    if isinstance(args[0], str):
        lp, hp = wavefilter(args[0], "d")
        if len(args) >= 2:
            extmode = args[1]
    else:
        if len(args) < 2:
            raise ValueError("LP and HP filters are required.")
        lp = args[0]
        hp = args[1]
        if len(args) >= 3:
            extmode = args[2]

    lp = np.asarray(lp).reshape(-1)
    hp = np.asarray(hp).reshape(-1)
    fl = len(lp)

    x = np.asarray(x)
    sx = x.shape
    if (
        x.ndim not in (2, 3)
        or min(sx[:2]) < 2
        or not np.isrealobj(x)
        or not np.issubdtype(x.dtype, np.number)
    ):
        raise ValueError("X must be a real, numeric 2-D or 3-D matrix.")

    if (
        lp.ndim != 1
        or hp.ndim != 1
        or not np.isrealobj(lp)
        or not np.isrealobj(hp)
        or not np.issubdtype(lp.dtype, np.number)
        or not np.issubdtype(hp.dtype, np.number)
        or fl != len(hp)
        or (fl % 2) != 0
    ):
        raise ValueError(
            "LP and HP must be even and equal length real, numeric filter vectors."
        )

    if not np.isscalar(n) or not np.isreal(n):
        raise ValueError("N must be a real scalar.")
    n = int(n)
    if n < 1 or n > np.log2(max(sx)):
        raise ValueError("N must be between 1 and log2(max(size(X))).")

    pages = 1 if x.ndim == 2 else x.shape[2]

    # Init output structures.
    c = np.array([], dtype=float)
    s = [list(sx[:2])]

    app = []
    for i in range(pages):
        app.append(double_page(x, i, pages))

    # For each decomposition level.
    for _ in range(n):
        app, keep = _extend(app, fl, pages, extmode)

        rows = _convolve(app, hp, "row", fl, keep, pages, extmode)
        coefs = _convolve(rows, hp, "col", fl, keep, pages, extmode)
        c = _addcoefs(c, coefs, pages)
        s.insert(0, list(coefs[0].shape))

        coefs = _convolve(rows, lp, "col", fl, keep, pages, extmode)
        c = _addcoefs(c, coefs, pages)

        rows = _convolve(app, lp, "row", fl, keep, pages, extmode)
        coefs = _convolve(rows, hp, "col", fl, keep, pages, extmode)
        c = _addcoefs(c, coefs, pages)

        app = _convolve(rows, lp, "col", fl, keep, pages, extmode)

    # Append final approximation.
    c = _addcoefs(c, app, pages)
    s.insert(0, list(app[0].shape))
    s = np.array(s, dtype=int)
    if x.ndim == 3:
        s = np.hstack([s, pages * np.ones((s.shape[0], 1), dtype=int)])

    return c, s


def double_page(x: Any, i: Any, pages: Any):
    """double_page."""
    if pages == 1:
        return x.astype(float)
    return x[:, :, i].astype(float)


def _addcoefs(c: Any, x: Any, pages: Any):
    """Add page coefficients to decomposition vector."""
    nc = c
    for i in range(pages - 1, -1, -1):
        nc = np.concatenate([x[i].reshape(-1, order="F"), nc])
    return nc


def _extend(x: Any, fl: Any, pages: Any, extmode: Any):
    """Extend each page in both dimensions and return keep sizes."""
    y = []
    keep = None
    for i in range(pages):
        if str(extmode).upper() == "SYM":
            keep = np.floor((fl + np.array(x[i].shape) - 1) / 2).astype(int)
            y.append(padarray(x[i], [fl - 1, fl - 1], "symmetric", "both"))
        elif str(extmode).upper() == "PER":
            keep = np.array(x[i].shape, dtype=int)
            y.append(padarray(x[i], [fl // 2, fl // 2], "circular", "both"))
        else:
            raise ValueError("Invalid extension mode!")
    return y, keep


def _convolve(
    x: Any, h: Any, axis_type: Any, fl: Any, keep: Any, pages: Any, extmode: Any
):
    """Convolve rows/columns, downsample, and crop by keep."""
    y = []
    h = np.asarray(h).reshape(-1)

    for i in range(pages):
        if axis_type == "row":
            if str(extmode).upper() == "SYM":
                tmp = convolve2d(x[i], h.reshape(1, -1), mode="full")
                tmp = tmp[:, 0::2]
                tmp = tmp[:, fl // 2 : fl // 2 + keep[1]]
            else:
                tmp = convolve2d(x[i], h.reshape(1, -1), mode="valid")
                tmp = tmp[:, 1 : 2 * int(np.ceil(keep[1] / 2)) : 2]
        else:
            if str(extmode).upper() == "SYM":
                tmp = convolve2d(x[i], h.reshape(-1, 1), mode="full")
                tmp = tmp[0::2, :]
                tmp = tmp[fl // 2 : fl // 2 + keep[0], :]
            else:
                tmp = convolve2d(x[i], h.reshape(-1, 1), mode="valid")
                tmp = tmp[1 : 2 * int(np.ceil(keep[0] / 2)) : 2, :]
        y.append(tmp)

    return y
