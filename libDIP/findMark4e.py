from typing import Any
import numpy as np
from scipy.fftpack import dct


def findMark4e(f: Any, w: Any):
    """
    Looks for the presence of invisible watermark W in image F and
    returns correlation coefficient C. C = 1 is perfect correlation.

    Parameters:
    -----------
    f : numpy.ndarray
        Image (possibly watermarked).
    w : dict
        Watermark structure containing:
        'index': indices of coefficients where watermark was applied.
        'coef': original coefficients (from the original image).
        'm': the watermark sequence inserted.

    Returns:
    --------
    c : float
        Correlation coefficient.
    """
    if not isinstance(w, dict):
        raise ValueError("w must be a dictionary.")

    f = np.array(f, dtype=float)

    # DCT2
    # scipy.fftpack.dct computes 1D DCT.
    # To compute 2D DCT, apply to rows then cols.
    # type=2 is standard. norm='ortho' makes it orthogonal (unitary).
    tfrm = dct(dct(f.T, norm="ortho").T, norm="ortho")

    # Note: Flattening order must match how indices were generated!
    # If the user generated indices using flattened 'C' order (row-major), we must flatten same way.
    # MATLAB uses 'F' order (column-major).
    # Since we are implementing the python ecosystem, we assume standard numpy 'C' order.
    # Ideally indices are tuple (x_coords, y_coords).
    # If 'index' is 1D array, we flatten tfrm.

    indices = w["index"]

    if isinstance(indices, tuple) and len(indices) == 2:
        # Tuple indexing (row_idx, col_idx)
        wtst = tfrm[indices]
    else:
        # Flat indexing
        wtst = tfrm.flatten()[indices]

    # Extract embedded mark
    # formula: wtst = coef * (1 + alpha * w)
    # So w_est = (wtst - coef) / (alpha * coef)
    # The function assumes alpha=0.1
    alpha = 0.1

    coef_orig = w["coef"]

    # Avoid division by zero
    divisor = alpha * coef_orig
    # Use safe division or assume coefficients are significant (watermarking usually targets high energy ones)

    # Estimate watermark sequence
    w_est = (wtst - coef_orig) / divisor

    # Compute correlation
    # corr2(A, B) computes 2D correlation coefficient.
    # Here inputs are 1D vectors.
    # np.corrcoef returns matrix.

    w_orig = w["m"]

    # Pearson correlation coefficient
    c_matrix = np.corrcoef(w_orig.flatten(), w_est.flatten())
    c = c_matrix[0, 1]

    return c
