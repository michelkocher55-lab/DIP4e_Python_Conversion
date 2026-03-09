from typing import Any
import numpy as np
from lib.morphoGeoDilate4e import morphoGeoDilate4e


def morphoReconDilate4e(F: Any, G: Any, B: Any = None):
    """
    Computes morphological reconstruction by dilation.
    Iteratively dilates F constrained by G until stability.

    RD, k = morphoReconDilate4e(F, G, B=None)

    Parameters
    ----------
    F : numpy.ndarray
        Marker image.
    G : numpy.ndarray
        Mask image.
    B : numpy.ndarray, optional
        Structuring element. Default 3x3 ones.

    Returns
    -------
    RD : numpy.ndarray
        Reconstructed image.
    k : int
        Number of iterations until stability.
    """

    if B is None:
        B = np.ones((3, 3))

    F = np.array(F)
    G = np.array(G)
    B = np.array(B)

    rdPrevious = F
    # First step
    RD = morphoGeoDilate4e(F, G, B, 1)
    k = 1

    while not np.array_equal(RD, rdPrevious):
        k += 1
        rdPrevious = RD
        RD = morphoGeoDilate4e(rdPrevious, G, B, 1)

    # k is the step where it became equal. So stability reached at k.
    # MATLAB code does k = k - 1?
    # Loop:
    # 1. k=1. Calc RD1.
    # 2. Check RD1 != F.
    # 3. k=2. prev=RD1. Calc RD2.
    # ...
    # k-th iteration finds no change.
    # The result was already stable at k-1?
    # Yes, if RD_k == RD_(k-1), then stability was reached at k-1 (the first time the value appeared).
    # MATLAB: k starts 1. Loop while different.
    # So if it terminates, k counts the step where NO change happened.
    # So stability (the value) was reached at k-1.

    return RD, k - 1
