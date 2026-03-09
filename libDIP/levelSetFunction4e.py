from typing import Any
import numpy as np
from scipy.ndimage import distance_transform_edt


def levelSetFunction4e(type_str: Any, *args: Any):
    """
    Generates a level-set function (Signed Distance Function).

    phi = levelsetFunction4e(type_str, *args)

    Parameters
    ----------
    type_str : str
        'mask' or 'circular'.
    args : variadic
        If type_str == 'mask':
            args[0] : binary mask (numpy.ndarray). 1 inside, 0 outside.
        If type_str == 'circular':
            args[0] : M (rows)
            args[1] : N (cols)
            args[2] : x0 (center row)
            args[3] : y0 (center col)
            args[4] : r (radius)

    Returns
    -------
    phi : numpy.ndarray
        Signed distance function. Negative inside, Positive outside.
    """

    type_str = type_str.lower()

    if type_str == "mask":
        binmask = args[0]
        binmask = (binmask > 0).astype(float)

        # distance_transform_edt computes distance to nearest zero (background)
        # Term 1: Distance outside object to object.
        # We invert mask: 1s outside, 0s inside. edt gives dist to object bounds.
        dist_outside = distance_transform_edt(1 - binmask)

        # Term 2: Distance inside object to background.
        # Mask: 1s inside, 0s outside. edt gives dist to background.
        dist_inside = distance_transform_edt(binmask)

        # Phi = DistOutside - DistInside
        phi = dist_outside - dist_inside + (binmask - 0.5)

    elif type_str == "circular":
        M, N, x0, y0, r = args
        # M is rows, N is cols.
        rr, cc = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")

        # Using linear SDF for stability
        # phi = np.sqrt((rr - x0)**2 + (cc - y0)**2) - r
        phi = (rr - x0) ** 2 + (cc - y0) ** 2 - r**2  # MKR

    else:
        raise ValueError("Type must be 'mask' or 'circular'.")

    return phi
