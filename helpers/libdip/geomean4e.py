import numpy as np
from scipy.ndimage import uniform_filter

def geomean4e(g, m, n):
    """
    Geometric mean spatial filter.
    
    Parameters:
    -----------
    g : numpy.ndarray
        Input image.
    m : int
        Row size of the filter.
    n : int
        Column size of the filter.
        
    Returns:
    --------
    f_hat : numpy.ndarray
        Filtered image.
    """
    g = np.array(g, dtype=float)
    
    # Epsilon to avoid log(0)
    eps = np.finfo(float).eps
    g = g + eps
    
    # Geometric mean = product(neighborhood)^(1/mn)
    #                = exp( sum(log(neighborhood)) / mn )
    #                = exp( mean(log(neighborhood)) )
    
    # Compute log
    log_g = np.log(g)
    
    # Compute mean of log
    mean_log = uniform_filter(log_g, size=(m, n), mode='reflect')
    
    # Exponentiate
    f_hat = np.exp(mean_log)
    
    return f_hat
