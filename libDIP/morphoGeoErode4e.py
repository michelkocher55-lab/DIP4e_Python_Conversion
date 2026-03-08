import numpy as np
from lib.morphoErode4e import morphoErode4e

def morphoGeoErode4e(F, G, B, n):
    """
    Computes binary geodesic erosion of size n.
    result = Erode(F, B) | G, iterated n times.
    The result stays a superset of G.
    
    eg = morphoGeoErode4e(F, G, B, n)
    
    Parameters
    ----------
    F : numpy.ndarray
        Marker binary image (Start set). F usually >= G.
    G : numpy.ndarray
        Mask binary image (Constraint).
    B : numpy.ndarray
        Structuring element.
    n : int
        Number of iterations.
        
    Returns
    -------
    eg : numpy.ndarray
        Geodesic erosion result.
    """
    
    eg = np.array(F)
    G_bool = (np.array(G) > 0)
    B = np.array(B)
    
    for _ in range(n):
        # 1. Erode
        eroded = morphoErode4e(eg, B)
        
        # 2. Union with Mask G
        # Boolean OR
        eg_next = (eroded > 0) | G_bool
        
        eg = eg_next.astype(float)
        
    return eg
