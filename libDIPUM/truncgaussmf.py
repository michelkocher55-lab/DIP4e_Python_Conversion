
import numpy as np

def truncgaussmf(z, a, b, s):
    """
    Truncated Gaussian membership function.
    
    Parameters:
    z (ndarray): Input variable.
    a (float): Start of the non-zero range.
    b (float): Center/Peak of the Gaussian. (Must be >= a).
    s (float): Standard deviation (sigma).
    
    Returns:
    mu (ndarray): Membership function values.
    
    Logic:
    mu = exp(-(z - b)^2 / (2*s^2))  if a <= z <= (2b - a)
    mu = 0                          otherwise
    """
    z = np.asarray(z, dtype=float)
    mu = np.zeros_like(z)
    
    # Calculate upper limit c
    # MATLAB: c = a + 2*(b - a);
    c = a + 2 * (b - a)
    
    # Range check
    # range = (a <= z) & (z <= c);
    mask = (z >= a) & (z <= c)
    
    # Calculate Gaussian
    if np.any(mask):
        mu[mask] = np.exp(-(z[mask] - b)**2 / (2 * s**2))
        
    return mu
