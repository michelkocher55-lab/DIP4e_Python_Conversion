from typing import Any
import numpy as np
from lib.morphoErode4e import morphoErode4e


def morphoBoundary4e(I: Any, B: Any = None):
    """
    Computes the boundary of objects in a binary image.

    BD = morphoBoundary4e(I, B=None)

    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray, optional
        Structuring element. Defaults to 3x3 ones.

    Returns
    -------
    BD : numpy.ndarray
        Boundary image.
    """

    I = np.array(I)

    # Default B
    if B is None:
        B = np.ones((3, 3))
    else:
        B = np.array(B)

    # Ensure I is binary logical for bitwise ops, or use 0/1 float logic.
    # morphoErode4e returns float 0/1 usually.
    # We will treat nonzero as True.

    E = morphoErode4e(I, B)

    # Boundary = I AND NOT E
    # I should be the foreground. E is the eroded foreground (smaller).
    # The difference is the rim.

    I_bool = I > 0
    E_bool = E > 0

    BD_bool = np.logical_and(I_bool, np.logical_not(E_bool))

    # Return as float 0.0/1.0 to match convention
    BD = BD_bool.astype(float)

    return BD
