from typing import Any
from lib.brFilterTF4e import brFilterTF4e


def bpFilterTF4e(filter_type: Any, M: Any, N: Any, C0: Any, W: Any, n: Any = 1):
    """
    Generates a bandpass filter transfer function.

    Parameters:
    -----------
    filter_type : str
        'ideal', 'gaussian', or 'butterworth'.
    M : int
        Number of rows.
    N : int
        Number of columns.
    C0 : float
        Cutoff frequency or center frequency of the band.
    W : float
        Width of the band.
    n : float, optional
        Order of the Butterworth filter. Default is 1.

    Returns:
    --------
    H : numpy.ndarray
        M x N transfer function.
    """

    # H_bp = 1 - H_br
    H_br = brFilterTF4e(filter_type, M, N, C0, W, n)
    H = 1.0 - H_br
    return H
