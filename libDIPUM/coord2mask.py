import numpy as np
from skimage.draw import polygon


def coord2mask(M, N, vx, vy):
    """
    Generates a binary mask from given coordinates.

    mask = coord2mask(M, N, vx, vy)
    """
    vx = np.asarray(vx)
    vy = np.asarray(vy)

    rr, cc = polygon(vy, vx, (M, N))
    mask = np.zeros((M, N), dtype=float)
    mask[rr, cc] = 1.0
    return mask
