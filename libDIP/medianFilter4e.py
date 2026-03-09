from typing import Any
import numpy as np
from scipy.ndimage import median_filter


def medianFilter4e(g: Any, m: Any, n: Any):
    """
    2-D median filter.

    Parameters:
    -----------
    g : numpy.ndarray
        Input image.
    m : int
        Height of the filter window.
    n : int
        Width of the filter window.

    Returns:
    --------
    f_hat : numpy.ndarray
        Filtered image.
    """
    g = np.array(g, dtype=float)

    # Use scipy.ndimage.median_filter
    # To match typical MATLAB 'nlfilter' padding (zero padding),
    # we use mode='constant', cval=0.0.

    f_hat = median_filter(g, size=(m, n), mode="constant", cval=0.0)

    return f_hat
