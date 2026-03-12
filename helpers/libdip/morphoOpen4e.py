import numpy as np
import warnings
from helpers.libdip.morphoErode4e import morphoErode4e
from helpers.libdip.morphoDilate4e import morphoDilate4e

def morphoOpen4e(I, B):
    """
    Computes morphological opening of binary image I using structuring element B.
    Opening is Erosion followed by Dilation.
    
    O = morphoOpen4e(I, B)
    
    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray
        Structuring element.
        
    Returns
    -------
    O : numpy.ndarray
        Opened image.
    """
    
    B = np.array(B)
    m, n = B.shape
    
    if np.sum(B) != m * n:
        warnings.warn("For opening (involving erosion), all elements of B should be 1; 0s can lead to unexpected results.")
    
    # Eq 9-10: Opening = Dilation(Erosion(I, B), B)
    
    # 1. Erode
    eroded = morphoErode4e(I, B)
    
    # 2. Dilate
    O = morphoDilate4e(eroded, B)
    
    return O
