import numpy as np
from libDIP.morphoMatch4e import morphoMatch4e

def morphoHitMiss4e(I, B, padval=0, mode='same'):
    """
    Computes morphological hit-miss transform of binary image.
    
    H = morphoHitMiss4e(I, B, padval=0, mode='same')
    
    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray
        Structuring element (0, 1, or Don't Care).
    padval : int
        Padding value.
    mode : str
        'same' or 'full'.
        
    Returns
    -------
    H : numpy.ndarray
        Hit-miss transform (binary).
    """
    
    # morphoMatch4e returns 1.0 for perfect match, 0.5 for partial, 0.0 for none.
    # Hit-Miss Transform requires PERFECT match of the specified foreground/background constraints in B.
    
    S = morphoMatch4e(I, B, padval=padval, mode=mode)
    
    # Only keep perfect matches
    H = (S == 1.0).astype(float)
    
    return H
