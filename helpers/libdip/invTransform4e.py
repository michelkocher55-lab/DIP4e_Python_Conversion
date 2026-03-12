import numpy as np

from helpers.libdip.tmat4e import tmat4e

def invTransform4e(t, xform):
    """
    Compute 1-D or 2-D inverse XFORM transform.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Transform coeffs.
    xform : str
        Transform name.
        
    Returns:
    --------
    f : numpy.ndarray
        Reconstructed signal.
    """
    t = np.array(t)
    # Check dimensions
    s = t.shape
    
    # 1D
    if t.ndim == 1:
        t = t.reshape(-1, 1)
        s = t.shape
        
    if s[1] == 1:
        # Vector
        N = s[0]
        a = tmat4e(xform, N)
        # f = transpose(conj(a)) * t
        f = a.conj().T @ t
        return f.flatten()
        
    elif s[0] == s[1]:
        # Square Matrix
        N = s[0]
        a = tmat4e(xform, N)
        # f = transpose(conj(a)) * t * conj(a) (MATLAB)
        # In python: a.conj().T @ t @ a.conj()
        f = a.conj().T @ t @ a.conj()
        return f
    else:
        raise ValueError("2-D transforms must be square!")
