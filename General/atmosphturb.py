from typing import Any
import numpy as np
from libDIPUM.dftuv import dftuv


def atmosphturb(M: Any, N: Any, k: Any):
    """
    Atmospheric turbulence transfer function.

    H = atmosphturb(M, N, k)

    Parameters:
        M, N: size of the transfer function
        k: turbulence constant

    Returns:
        H: transfer function (uncentered)
    """
    # Use dftuv to set up frequency grid
    U, V = dftuv(M, N)

    # Distances
    D = np.hypot(U, V)

    # Filter
    H = np.exp(-k * ((D**2) ** (5.0 / 6.0)))
    H = np.fft.ifftshift(H)

    return H
