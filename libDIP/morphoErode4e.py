from typing import Any
import numpy as np
import warnings
from lib.morphoMatch4e import morphoMatch4e


def morphoErode4e(I: Any, B: Any, padval: Any = 0):
    """
    Computes morphological erosion of binary image I using structuring element B.

    E = morphoErode4e(I, B, padval=0)

    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray
        Structuring element (usually all 1s).
    padval : int
        Padding value (0 or 1).

    Returns
    -------
    E : numpy.ndarray
        Eroded image (0s and 1s).
    """

    B = np.array(B)
    m, n = B.shape

    if np.sum(B) != m * n:
        warnings.warn(
            "For erosion, all elements of B should be 1; 0s can lead to unexpected results."
        )

    # Use morphoMatch4e to find perfect matches
    # Perfect match (1.0) implies B is fully contained in foreground of I.
    S = morphoMatch4e(I, B, padval=padval, mode="same")

    # Erosion is 1 where S == 1 (Perfect Match), 0 otherwise.
    E = (S == 1.0).astype(float)

    return E
