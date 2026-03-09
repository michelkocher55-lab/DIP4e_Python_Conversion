from typing import Any
import numpy as np
import warnings
from lib.morphoErode4e import morphoErode4e


def morphoDilate4e(I: Any, B: Any):
    """
    Computes morphological dilation of binary image I using structuring element B.

    D = morphoDilate4e(I, B)

    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray
        Structuring element (usually all 1s, odd dims).

    Returns
    -------
    D : numpy.ndarray
        Dilated image.
    """

    I = np.array(I)
    B = np.array(B)

    m, n = B.shape

    if np.sum(B) != m * n:
        warnings.warn(
            "For dilation, all elements of B should be 1; 0s can lead to unexpected results."
        )

    # Duality: Dilation(I, B) = NOT(Erosion(NOT(I), Reflect(B)))

    # 1. Complement of I
    # Assuming I is 0/1, logical_not works or 1-I.
    # Ensure I is binary 0/1 float/int for safety
    I_bin = I > 0
    Ic = np.logical_not(I_bin).astype(float)

    # 2. Reflect B
    Bhat = np.rot90(B, 2)

    # 3. Erode Ic with Bhat
    # Padval must be 1 because background of Ic corresponds to foreground of I (usually 0).
    # Wait, background of Ic is 1 (since I background is 0).
    # So we pad with 1.
    E_temp = morphoErode4e(Ic, Bhat, padval=1)

    # 4. Complement of Erosion
    D = np.logical_not(E_temp > 0).astype(float)

    return D
