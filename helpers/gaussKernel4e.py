from typing import Any
import numpy as np


def gaussKernel4e(m: Any, sig: Any, K: Any = 1):
    """
    Generates a Gaussian kernel.

    Parameters:
        m: Size (must be odd).
        sig: Sigma.
        K: Scale factor (defaults to 1).

    Returns:
        w: Normalized Gaussian kernel.
    """
    # Check odd
    m = int(m)
    if m % 2 == 0:
        raise ValueError("The dimensions of the neighborhood must be odd")

    # Coordinates
    limit = (m - 1) / 2
    x = np.linspace(-limit, limit, m)
    y = np.linspace(-limit, limit, m)
    X, Y = np.meshgrid(x, y)

    # Kernel
    w = K * np.exp(-(X**2 + Y**2) / (2 * sig**2))

    # Normalize
    w_sum = np.sum(w)
    if w_sum != 0:
        w = w / w_sum

    return w
