from typing import Any
import numpy as np
import warnings
from lib.morphoErode4e import morphoErode4e
from lib.morphoDilate4e import morphoDilate4e


def morphoClose4e(I: Any, B: Any):
    """
    Computes morphological closing of binary image I using structuring element B.
    Closing is Dilation followed by Erosion.

    C = morphoClose4e(I, B)

    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray
        Structuring element (usually all 1s).

    Returns
    -------
    C : numpy.ndarray
        Closed image.
    """

    B = np.array(B)
    m, n = B.shape

    if np.sum(B) != m * n:
        warnings.warn(
            "For closing (involving erosion), all elements of B should be 1; 0s can lead to unexpected results."
        )

    # Eq 9-11: Closing = Erosion(Dilation(I, B), B)

    # 1. Dilate
    dilated = morphoDilate4e(I, B)

    # 2. Erode
    C = morphoErode4e(dilated, B)

    return C
