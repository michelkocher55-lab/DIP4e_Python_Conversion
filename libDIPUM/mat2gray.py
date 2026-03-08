
import numpy as np

def mat2gray(f, limits=None):
    """
    Converts matrix to grayscale image (scales to 0-1).
    Mirroring MATLAB's mat2gray.
    
    g = mat2gray(A, [amin amax]) maps values in A to range [0, 1].
    Values < amin overlap to 0, > amax overlap to 1.
    
    If limits is None, min and max of A are used.
    """
    f = np.asarray(f, dtype=float)
    
    if limits is None:
        min_val = f.min()
        max_val = f.max()
    else:
        min_val, max_val = limits
        
    if max_val == min_val:
        # Avoid division by zero, return 0 or 1?
        # MATLAB behavior: If image is constant, mat2gray returns 0s (if min!=max logic fails) or handled?
        # Typically returns 0 if constant.
        return np.zeros_like(f)
        
    g = (f - min_val) / (max_val - min_val)
    g = np.clip(g, 0.0, 1.0)
    
    return g
