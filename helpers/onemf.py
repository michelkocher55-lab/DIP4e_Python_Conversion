from typing import Any
import numpy as np


def onemf(z: Any, *args: Any):
    """
    Constant membership function (one).

    Parameters:
    z (ndarray): Input variable.
    *args: Ignored additional arguments (to match signature of other MFs).

    Returns:
    mu (ndarray): Array of ones with same size as z.
    """
    z = np.asarray(z)
    return np.ones_like(z, dtype=float)
