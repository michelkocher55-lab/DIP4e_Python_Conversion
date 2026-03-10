from typing import Any
import numpy as np


def zeromf(z: Any, *args: Any):
    """
    Constant membership function (zero).

    Parameters:
    z (ndarray): Input variable.
    *args: Ignored additional arguments (to match signature of other MFs).

    Returns:
    mu (ndarray): Array of zeros with same size as z.
    """
    z = np.asarray(z)
    return np.zeros_like(z, dtype=float)
