import numpy as np
import scipy.ndimage

def minFilter4e(g, m, n):
    """
    2-D min filter.
    Applies a 2-D min filter of size m-by-n to image g.
    
    Parameters:
    -----------
    g : numpy.ndarray
        Input image.
    m : int
        Height of neighborhood (rows).
    n : int
        Width of neighborhood (cols).
        
    Returns:
    --------
    f_hat : numpy.ndarray
        Filtered image.
    """
    g = g.astype(float)
    
    # Apply minimum filter
    # mode='reflect' is used to avoid zero-padding artifacts (black borders) 
    # which would occur with strict zero padding if g > 0.
    f_hat = scipy.ndimage.minimum_filter(g, size=(m, n), mode='reflect')
    
    return f_hat
