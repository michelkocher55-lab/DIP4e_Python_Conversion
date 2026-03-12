import numpy as np
from helpers.libdip.morphoReconErode4e import morphoReconErode4e

def morphoClosebyRecon4e(marker, mask):
    """
    Computes morphological closing by reconstruction.
    Wrapper for morphoReconErode4e(marker, mask, ones(3,3)).
    
    To perform "Closing by Reconstruction" on Image I:
        marker = morphoDilate4e(I, B_dilate)
        mask = I
        CR, k = morphoClosebyRecon4e(marker, mask)
        
    CR, k = morphoClosebyRecon4e(marker, mask)
    
    Parameters
    ----------
    marker : numpy.ndarray
        Marker image (e.g. Dilation of I).
    mask : numpy.ndarray
        Mask image (e.g. I).
        
    Returns
    -------
    CR : numpy.ndarray
        Result.
    k : int
        Iterations.
    """
    
    B = np.ones((3, 3))
    return morphoReconErode4e(marker, mask, B)
