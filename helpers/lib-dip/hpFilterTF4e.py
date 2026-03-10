import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from libDIP.lpFilterTF4e import lpFilterTF4e

def hpFilterTF4e(filter_type, P, Q, D0, n=1):
    """
    Circularly-symmetric highpass filter transfer function.
    
    Parameters:
    -----------
    filter_type : str
        'ideal', 'gaussian', or 'butterworth'.
    P : int
        Number of rows.
    Q : int
        Number of columns.
    D0 : float
        Cutoff frequency.
    n : float, optional
        Order of the Butterworth filter. Default is 1.
        
    Returns:
    --------
    H : numpy.ndarray
        P x Q transfer function.
    """
    
    # Generate lowpass filter
    H_lp = lpFilterTF4e(filter_type, P, Q, D0, n)
    
    # Generate highpass from lowpass: HP = 1 - LP
    H = 1.0 - H_lp
    
    return H
