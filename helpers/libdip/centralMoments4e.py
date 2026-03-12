import numpy as np
from helpers.libdip.imageHist4e import imageHist4e

def centralMoments4e(f, n):
    """
    Computes central moments.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image (8-bit grayscale, [0,1] or [0,255]).
    n : int
        Number of moments to compute.
        
    Returns:
    --------
    u : numpy.ndarray
        Array of size n containing moments.
        u[0] is the mean.
        u[1] is the variance (2nd central moment).
        u[2] is the 3rd central moment, etc.
    """
    
    # Compute normalized histogram
    p = imageHist4e(f, mode='n')
    
    f = np.array(f, dtype=float)
    
    # Determine z values
    if f.size > 0 and np.max(f) <= 1:
        # Range [0, 1] mapped to 256 bins
        z = np.arange(256) / 255.0
    else:
        # Range [0, 255]
        z = np.arange(256).astype(float)
        
    u = np.zeros(n)
    
    # Mean
    m = np.sum(z * p)
    u[0] = m
    
    # Higher moments
    # MATLAB loop: for I = 2:n
    # Python indices: 1 to n-1
    # Check if n >= 2
    
    for i in range(2, n + 1):
        # i is the order (2, 3... n)
        # u index is i-1
        moment = np.sum( ((z - m)**i) * p )
        u[i-1] = moment
        
    return u
