from typing import Any
import numpy as np
from lib.morphoDilate4e import morphoDilate4e


def morphoGeoDilate4e(F: Any, G: Any, B: Any, n: Any):
    """
    Computes binary geodesic dilation of size n.
    result = Dilate(F, B) & G, iterated n times.

    dg = morphoGeoDilate4e(F, G, B, n)

    Parameters
    ----------
    F : numpy.ndarray
        Marker binary image.
    G : numpy.ndarray
        Mask binary image.
    B : numpy.ndarray
        Structuring element (usually 3x3 ones).
    n : int
        Number of iterations (size).

    Returns
    -------
    dg : numpy.ndarray
        Geodesic dilation result.
    """

    dg = np.array(F)
    G = np.array(G)
    B = np.array(B)

    # Ensure binary logic
    G_bool = G > 0

    # Iterate n times
    for k in range(n):
        # 1. Dilate current
        dilated = morphoDilate4e(dg, B)

        # 2. Intersect with Mask G
        # Boolean AND
        dilated_masked = (dilated > 0) & G_bool

        dg = dilated_masked.astype(float)

    return dg
