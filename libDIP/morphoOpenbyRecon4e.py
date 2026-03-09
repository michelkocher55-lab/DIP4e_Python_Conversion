from typing import Any
import numpy as np
from lib.morphoReconDilate4e import morphoReconDilate4e


def morphoOpenbyRecon4e(marker: Any, mask: Any):
    """
    Computes morphological opening by reconstruction.
    Actually, this function just performs Morphological Reconstruction by Dilation
    of 'marker' into 'mask' using a 3x3 structuring element.

    To perform a full "Opening by Reconstruction" on image I with SE B_erode:
       marker = morphoErode4e(I, B_erode)
       mask = I
       OR, k = morphoOpenbyRecon4e(marker, mask)

    OR, k = morphoOpenbyRecon4e(marker, mask)

    Parameters
    ----------
    marker : numpy.ndarray
        Marker image (e.g. results of erosion).
    mask : numpy.ndarray
        Mask image (e.g. original image).

    Returns
    -------
    OR : numpy.ndarray
        Result.
    k : int
        Iterations.
    """

    # Defaults B to ones((3,3)) inside morphoReconDilate4e if not passed?
    # But here we pass ones((3,3)) explicitly to match MATLAB code.
    B = np.ones((3, 3))

    return morphoReconDilate4e(marker, mask, B)
