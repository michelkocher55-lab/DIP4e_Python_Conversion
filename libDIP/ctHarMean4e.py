from typing import Any
import numpy as np
from scipy.ndimage import uniform_filter


def ctHarMean4e(g: Any, m: Any, n: Any, q: Any):
    """
    Contraharmonic mean spatial filter.

    Parameters:
    -----------
    g : numpy.ndarray
        Input image.
    m : int
        Row size of the filter.
    n : int
        Column size of the filter.
    q : float
        Order of the filter.
        Q > 0 eliminates pepper noise (dark).
        Q < 0 eliminates salt noise (light).

    Returns:
    --------
    f_hat : numpy.ndarray
        Filtered image.
    """
    g_norm = np.array(g, dtype=float)

    # Epsilon to avoid division by zero
    eps = np.finfo(float).eps

    # Handle singularity for negative Q if g contains 0
    # Add eps to base to ensure stability.
    g_norm = g_norm + eps

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            term_num = g_norm ** (q + 1)
            term_den = g_norm**q

    except ValueError:
        pass

    # Compute local means
    f_hatn = uniform_filter(term_num, size=(m, n), mode="reflect")
    f_hatd = uniform_filter(term_den, size=(m, n), mode="reflect")

    # Result
    # Avoid zero division in final division
    with np.errstate(divide="ignore", invalid="ignore"):
        f_hat = f_hatn / (f_hatd + eps)

    # Handle NaNs/Infs usually by clipping or setting to 0/1?
    # MATLAB code: f_hat = intScaling4e(f_hat, 'full');
    # This implies re-scaling min-max to [0, 1].

    f_hat = np.nan_to_num(f_hat)

    # Clip to valid range [0, 1] before scaling?
    # Or just scale min-max.
    # If we had division by zero, we might have huge values.
    # Let's clean up.

    # Re-scale to [0, 1]
    f_min = f_hat.min()
    f_max = f_hat.max()
    if f_max > f_min:
        f_hat = (f_hat - f_min) / (f_max - f_min)
    else:
        f_hat = f_hat - f_min

    return f_hat
