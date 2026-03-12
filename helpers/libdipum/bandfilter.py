from typing import Any
import numpy as np

try:
    from helpers.libdipum.dftuv import dftuv
except ImportError:
    try:
        from .dftuv import dftuv
    except ImportError:
        pass


def bandfilter(
    type_filter: Any, band: Any, M: Any, N: Any, C0: Any, W: Any, n: Any = 1
):
    """
    Computes frequency domain band filter transfer functions.

    H = bandfilter('ideal'|'butterworth'|'gaussian', band, M, N, C0, W, n)

    Returns uncentered transfer function (use fftshift for display).
    """
    U, V = dftuv(M, N)
    D = np.hypot(U, V)

    type_filter = type_filter.lower()
    band = band.lower()

    if type_filter == "ideal":
        H = _ideal_reject(D, C0, W)
    elif type_filter == "butterworth":
        H = _butterworth_reject(D, C0, W, n)
    elif type_filter == "gaussian":
        H = _gauss_reject(D, C0, W)
    else:
        raise ValueError("Unknown filter type.")

    if band == "pass":
        H = 1 - H
    elif band != "reject":
        raise ValueError("Unknown band type. Use 'reject' or 'pass'.")

    return H.astype(float)


def _ideal_reject(D: Any, C0: Any, W: Any):
    """_ideal_reject."""
    RI = D <= C0 - (W / 2.0)
    RO = D >= C0 + (W / 2.0)
    return (RO | RI).astype(float)


def _butterworth_reject(D: Any, C0: Any, W: Any, n: Any):
    """_butterworth_reject."""
    return 1.0 / (1.0 + ((D * W) / (D**2 - C0**2)) ** (2 * n))


def _gauss_reject(D: Any, C0: Any, W: Any):
    """_gauss_reject."""
    eps = np.finfo(float).eps
    return 1.0 - np.exp(-(((D**2 - C0**2) / (D * W + eps)) ** 2))
