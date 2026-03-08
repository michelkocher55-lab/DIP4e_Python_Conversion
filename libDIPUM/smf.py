
import numpy as np

def smf(z, a, b):
    """
    S-shaped membership function.
    
    Parameters:
    z (ndarray or scalar): Input variable.
    a (float): Lower bound (mu=0).
    b (float): Upper bound (mu=1).
    
    Returns:
    mu (ndarray): Membership values.
    """
    z = np.asarray(z)
    mu = np.zeros_like(z, dtype=float)
    
    p = (a + b) / 2.0
    
    # Cases
    # z < a: 0 (default initialized)
    
    # a <= z < p
    low_range = (z >= a) & (z < p)
    mu[low_range] = 2 * ((z[low_range] - a) / (b - a))**2
    
    # p <= z < b
    mid_range = (z >= p) & (z < b)
    mu[mid_range] = 1 - 2 * ((z[mid_range] - b) / (b - a))**2
    
    # b <= z
    high_range = (z >= b)
    mu[high_range] = 1.0
    
    return mu
