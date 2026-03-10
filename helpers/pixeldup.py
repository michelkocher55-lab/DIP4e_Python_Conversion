from typing import Any
import numpy as np


def pixeldup(A: Any, m: Any, n: Any = None):
    """
    Duplicates pixels of an image in both directions.

    Parameters:
    A : ndarray
        Input image.
    m : int
        Number of times to duplicate each pixel vertically.
    n : int (optional)
        Number of times to duplicate each pixel horizontally.
        If None, defaults to m.

    Returns:
    B : ndarray
        The upscaled image with size (m*rows, n*cols).
    """
    if n is None:
        n = m

    m = int(round(m))
    n = int(round(n))

    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")

    A = np.asarray(A)

    # Duplicate rows (vertical)
    if m > 1:
        B = np.repeat(A, m, axis=0)
    else:
        B = A

    # Duplicate cols (horizontal)
    if n > 1:
        B = np.repeat(B, n, axis=1)

    return B
