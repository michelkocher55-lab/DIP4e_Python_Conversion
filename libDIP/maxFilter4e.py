from typing import Any
import numpy as np
from scipy.ndimage import generic_filter


def maxFilter4e(g: Any, m: Any, n: Any):
    """
    2-D max filter.

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

    # Use generic_filter with np.max
    # MATLAB nlfilter uses zero padding ('constant', 0) efficiently?
    # Or implicitly zeros outside.
    # scipy generic_filter defaults to 'reflect'.
    # We choose 'constant' with cval=0.0 to match typical MATLAB assumption
    # (though explicit 'symmetric' is sometimes used, nlfilter docs mention padding with 0).

    # Note: generic_filter passes a 1D buffer to the function. np.max works on 1D.
    f_hat = generic_filter(g, np.max, size=(m, n), mode="constant", cval=0.0)

    return f_hat
