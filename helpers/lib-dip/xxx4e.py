import numpy as np
from lib.intScaling4e import intScaling4e
from lib.twodConv4e import twodConv4e

def xxx4e(g, m, n):
    """
    Arithmetic mean spatial filter.

    Parameters:
    -----------
    g : numpy.ndarray
        Input image.
    m : int
        Height of the filter.
    n : int
        Width of the filter.

    Returns:
    --------
    f_hat : numpy.ndarray
        Filtered image, scaled to [0, 1].
    """
    # Scale the input image to the range [0, 1].
    g = intScaling4e(g)

    # Define filter kernel.
    # Note: MATLAB code used (1/m*n)*ones(m,n) which might be interpreted as (n/m)*ones.
    # Standard arithmetic mean uses 1/(m*n).
    w = np.ones((m, n)) / (m * n)

    # Perform filtering.
    f_hat = twodConv4e(g, w)

    # Scale result to the full interval [0, 1].
    f_hat = intScaling4e(f_hat, mode='full')

    return f_hat
