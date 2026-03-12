from typing import Any

try:
    from helpers.libdipum.lpfilter import lpfilter
except ImportError:
    try:
        from .lpfilter import lpfilter
    except ImportError:
        pass


def hpfilter(type_filter: Any, M: Any, N: Any, D0: Any, n: Any = 1):
    """
    Computes frequency domain highpass filter transfer functions.

    Parameters:
    type_filter (str): 'ideal', 'butterworth', 'gaussian'.
    M, N (int): Size of the filter.
    D0 (float): Cutoff frequency.
    n (float, optional): Order for Butterworth filter. Default is 1.

    Returns:
    H (ndarray): Uncentered transfer function of size MxN.
    """

    # Hhp = 1 - Hlp
    Hlp = lpfilter(type_filter, M, N, D0, n)
    H = 1.0 - Hlp

    return H
