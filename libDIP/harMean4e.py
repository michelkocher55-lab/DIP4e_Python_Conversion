import numpy as np
import sys
import os

# Ensure we can import modules from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from twodConv4e import twodConv4e
from intScaling4e import intScaling4e

def harMean4e(g, m, n):
    """
    Harmonic Mean Spatial Filter.
    
    Parameters:
    -----------
    g : numpy.ndarray
        Input image.
    m : int
        Kernel height.
    n : int
        Kernel width.
        
    Returns:
    --------
    f_hat : numpy.ndarray
        Filtered image.
    """
    # Scale image to [0, 1]
    g = intScaling4e(g)
    
    # Kernel to perform sum
    w = np.ones((m, n))
    
    # Perform filtering
    # f_hat = numerator / denominator
    # denominator: convolution of 1/(g+eps) with w.
    
    term = 1.0 / (g + np.finfo(float).eps)
    
    # twodConv4e performs convolution.
    # Note: MATLAB code passes w directly. twodConv4e mimics MATLAB.
    conv_result = twodConv4e(term, w, param='ns') # 'ns' for no extra scaling inside conv
    
    f_hat = (m * n) / (conv_result + np.finfo(float).eps)
    
    # Scale result to full interval [0, 1]
    f_hat = intScaling4e(f_hat, mode='full')
    
    return f_hat
