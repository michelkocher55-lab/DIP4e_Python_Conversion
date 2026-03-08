
import numpy as np
from .trapezmf import trapezmf

def sigmamf(z, a, b):
    """
    Sigma membership function.
    Linear transition from 0 to 1 between a and b.
    
    Parameters:
    z (ndarray): Input.
    a, b (float): Parameters (a <= b).
    
    Returns:
    mu (ndarray): Membership.
    """
    # Defined as trapezmf(z, a, b, Inf, Inf) in MATLAB code.
    return trapezmf(z, a, b, float('inf'), float('inf'))
