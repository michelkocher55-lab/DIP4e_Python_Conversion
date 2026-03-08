import numpy as np
import sys
import os

# Ensure import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.tmat4e import tmat4e

def transform4e(f, xform):
    """
    Compute 1-D or 2-D XFORM transform of vector or matrix f.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input (N vector or NxN matrix).
    xform : str
        Transform name (e.g. 'DFT').
        
    Returns:
    --------
    t : numpy.ndarray
        Transform coeffs.
    """
    f = np.array(f, dtype=float) 
    # Handle complex inputs if DFT?
    if 'DFT' in xform.upper():
         f = f.astype(complex)
         
    s = f.shape
    
    # 1D Row Vector -> Transpose to column
    if f.ndim == 1:
        f = f.reshape(-1, 1)
        s = f.shape
        
    if s[1] == 1:
        # Vector
        N = s[0]
        # Check if row vector input logic needed? 
        # MATLAB: if s(1)==1, f=f'. Matrix logic.
        
        A = tmat4e(xform, N)
        t = A @ f
        return t.flatten() # Return vector
        
    elif s[0] == s[1]:
        # Square Matrix
        N = s[0]
        A = tmat4e(xform, N)
        t = A @ f @ A.T # Note: A for some transforms is real. For DFT A is complex.
        # MATLAB: A*f*transpose(A).
        # In Python @ is matmul.
        # transpose(A) in MATLAB is non-conjugate transpose.
        # numpy .T is non-conjugate transpose.
        # A.conj().T is conjugate.
        # MATLAB 'transpose' is .' -> non-conjugate.
        # MATLAB A is complex for DFT.
        # transform4e.m line 25: A*f*transpose(A).
        # So we use A.T
        return t
    else:
        raise ValueError("All 2-D inputs must be square!")
