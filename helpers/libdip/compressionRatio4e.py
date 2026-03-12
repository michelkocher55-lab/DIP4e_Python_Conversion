import os
import sys
import numpy as np

def compressionRatio4e(f, fc):
    """
    Computes the ratio of the bytes in two images/variables.
    
    Parameters:
    -----------
    f : str or object
        Original image/variable (uncompressed). 
        Can be a file path (str) or a Python object (e.g. numpy array).
    fc : str or object
        Compressed image/variable.
        Can be a file path (str) or a Python object.
        
    Returns:
    --------
    cr : float
        Compression ratio (bytes(f) / bytes(fc)).
    """
    
    def get_bytes(x):
        if isinstance(x, str):
            if os.path.exists(x):
                return os.path.getsize(x)
            else:
                # If string is not a file, measure string size?
                # MATLAB code assumes it's a filename if char.
                # If file doesn't exist, we should probably error or return sys.getsizeof
                raise ValueError(f"File not found: {x}")
        elif isinstance(x, np.ndarray):
            return x.nbytes
        else:
            return sys.getsizeof(x)
            
    b_f = get_bytes(f)
    b_fc = get_bytes(fc)
    
    if b_fc == 0:
        raise ValueError("Compressed size is 0.")
        
    cr = b_f / b_fc
    return cr
