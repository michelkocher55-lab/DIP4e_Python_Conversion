from typing import Any
import numpy as np


def mask4e(M: Any, N: Any, rUL: Any, cUL: Any, rLR: Any, cLR: Any):
    """
    Creates a binary image mask.

    R = mask4e(M, N, rUL, cUL, rLR, cLR) creates a binary image mask of
    size (M, N) with 1's in a rectangular region defined by upper
    left row and column coordinates at (rUL, cUL) and lower right
    coordinates at (rLR, cLR), and 0's elsewhere.

    Parameters:
    -----------
    M : int
        Number of rows.
    N : int
        Number of columns.
    rUL : int
        Upper-left row index (0-based, inclusive).
    cUL : int
        Upper-left column index (0-based, inclusive).
    rLR : int
        Lower-right row index (0-based, inclusive).
    cLR : int
        Lower-right column index (0-based, inclusive).

    Returns:
    --------
    R : numpy.ndarray
        Binary mask (float type with 0.0 and 1.0).
    """

    # Check dimensions
    if rUL < 0 or cUL < 0 or rLR >= M or cLR >= N:
        raise ValueError("Specified region of 1s exceeds M-by-N dimensions")

    R = np.zeros((M, N))

    # Python slicing: start is inclusive, end is exclusive.
    # So we go from rUL to rLR + 1
    R[rUL : rLR + 1, cUL : cLR + 1] = 1.0

    return R
