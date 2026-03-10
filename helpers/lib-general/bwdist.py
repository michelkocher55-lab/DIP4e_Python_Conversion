import numpy as np
from scipy.ndimage import distance_transform_edt
def bwdist(BW):
    """
    Computes the distance transform of a binary image.
    Distance to the nearest non-zero pixel.
    Mimics MATLAB's bwdist(BW).
    """
    # Invert the image: Calculate distance to valid object pixels (nearest True)
    # distance_transform_edt calculates distance to nearest 0 (False).
    # So we pass (BW == 0) to measure distance to the nearest 1 (True).
    return distance_transform_edt(BW == 0)