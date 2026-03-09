from typing import Any
import numpy as np
from lib.imageScaling4e import imageScaling4e


def imageResize4e(f: Any, v_size: Any, h_size: Any):
    """
    Resizes a grayscale or RGB image to new dimensions.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    v_size : int
        New vertical size (number of rows).
    h_size : int
        New horizontal size (number of columns).

    Returns:
    --------
    g : numpy.ndarray
        Resized image.
    """
    f = np.array(f)

    # Determine dimensions
    M, N = f.shape[:2]

    if M == 0 or N == 0:
        raise ValueError("Input dimensions must be non-zero.")

    # Calculate scale factors
    v_scale = v_size / M
    h_scale = h_size / N

    # Call imageScaling4e
    # imageScaling4e handles both grayscale and RGB if implemented to support 3D.
    # My implementation of imageScaling4e handles 3D arrays.
    g = imageScaling4e(f, v_scale, h_scale)

    return g
