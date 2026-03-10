from typing import Any
import numpy as np
from scipy.ndimage import convolve


def lap(f: Any):
    """
    L = lap(f)

    Compute the Laplacian of grayscale input image f.
    Output is floating-point (double precision by default).
    """
    # MATLAB: f = im2double(f)
    f = np.asarray(f)
    if np.issubdtype(f.dtype, np.integer):
        info = np.iinfo(f.dtype)
        f = f.astype(np.float64) / float(info.max)
    else:
        f = f.astype(np.float64)

    # Laplacian filter.
    h = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)

    # MATLAB: imfilter(..., 'replicate')
    L = convolve(f, h, mode="nearest")
    return L
