from typing import Any
import numpy as np


def gaussKernel4e(m: Any, sig: Any, K: Any = 1):
    """
    Circularly-symmetric lowpass Gaussian kernel.

    Parameters:
    -----------
    m : int
        Size of the kernel (must be odd).
    sig : float
        Standard deviation of the Gaussian.
    K : float, optional
         Scale factor for the Gaussian function (before normalization). Defaults to 1.

    Returns:
    --------
    w : numpy.ndarray
        MxM Gaussian kernel, normalized so sum is 1.
    """
    # Check if m is odd
    if m % 2 == 0:
        raise ValueError("The dimensions of the neighborhood must be odd")

    # Range of coordinates
    limit = (m - 1) // 2
    x = np.arange(-limit, limit + 1)
    y = np.arange(-limit, limit + 1)

    # Meshgrid
    # MATLAB: meshgrid(X, Y) -> X varies across cols, Y across rows
    # numpy: meshgrid(x, y) -> same default 'xy' indexing
    xx, yy = np.meshgrid(x, y)

    # Compute Gaussian
    # G(x, y) = K * exp(-(x^2 + y^2) / (2 * sig^2))
    w = K * np.exp(-(xx**2 + yy**2) / (2 * sig**2))

    # Normalize
    w_sum = np.sum(w)
    if w_sum != 0:
        w = w / w_sum

    return w
