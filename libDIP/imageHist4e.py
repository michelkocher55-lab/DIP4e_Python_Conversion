from typing import Any
import numpy as np


def imageHist4e(f: Any, mode: Any = "n"):
    """
    Computes histogram of an input image.

    Parameters:
    - f: Input image (array-like). Assumed to be grayscale.
         If max(f) <= 1, assumed to be in [0, 1] range and mapped to [0, 255].
         If max(f) > 1, assumed to be integer-like in [0, 255].
    - mode: 'n' for normalized (default), 'u' for unnormalized.

    Returns:
    - h: Histogram with 256 bins.
    """
    f = np.array(f, dtype=np.float64)

    if np.any(f < 0):
        raise ValueError("All image intensities must be positive")

    # Check range
    if np.max(f) <= 1.0:
        f = np.round(255 * f)

    f = f.astype(np.int64)

    # Histogram
    hist, _ = np.histogram(f, bins=256, range=(0, 256))

    if mode.lower() == "n":
        num_pixels = f.size
        h = hist / num_pixels
    elif mode.lower() == "u":
        h = hist
    else:
        h = hist

    return h
