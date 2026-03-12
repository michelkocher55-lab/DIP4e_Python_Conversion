import numpy as np
from helpers.libdip.morphoGeoErode4e import morphoGeoErode4e

def morphoReconErode4e(F, G, B=None):
    """
    Computes morphological reconstruction by erosion.
    Iteratively erodes F constrained by G (Union G) until stability.
    F is typically >= G. Result shrinks F down to G's "support" or features.
    
    RE, k = morphoReconErode4e(F, G, B=None)
    
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
    RE : numpy.ndarray
        Reconstructed image.
    k : int
        Number of iterations until stability.
    """
    
    if B is None:
        B = np.ones((3, 3))
        
    F = np.array(F)
    G = np.array(G)
    B = np.array(B)
    
    rePrevious = F
    # First step
    RE = morphoGeoErode4e(F, G, B, 1)
    k = 1
    
    while not np.array_equal(RE, rePrevious):
        k += 1
        rePrevious = RE
        RE = morphoGeoErode4e(rePrevious, G, B, 1)
        
    return RE, k - 1
