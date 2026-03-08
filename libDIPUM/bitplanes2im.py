
import numpy as np

def bitplanes2im(B, P=None, S=None):
    """
    Constructs an image from bit planes.
    
    Parameters:
    B (ndarray): Bit planes array of shape (M, N, n).
                 Assumes B[:,:,0] is Least Significant Bit (LSB).
                 Assumes B[:,:,n-1] is Most Significant Bit (MSB).
    P (list/ndarray, optional): Indices of planes to use (1-based to match description, or 0-based? 
                                Let's use 1-based indices for P to match typical MATLAB-like usage 
                                implied by the problem/transcode, OR standard Python 0-based.
                                
                                The MATLAB function uses 1-based P. "integers in range [1, n]".
                                To be Pythonic, I should probably use 0-based, BUT if I want to match
                                the logic exactly or allow direct porting of calls, 1-based might be safer?
                                No, Python users expect 0-based.
                                I will simply NOTE this. 
                                Let's support 1-based inputs if they look like MATLAB indices?
                                No, that's ambiguous. 
                                Decision: Use 0-based indices for P and S in Python.
                                P=[0, 1, 7] means planes 0 (LSB), 1, and 7 (MSB of 8).
    S (list/ndarray, optional): Shuffle positions (0-based).
                                Determines the "power of 2" position for the plane.
                                
    Returns:
    f (ndarray): Reconstructed image, float in range [0, 1].
    """
    
    B = np.asarray(B, dtype=float)
    if B.ndim != 3:
        raise ValueError("B must be 3-dimensional (M, N, n).")
        
    M, N, n = B.shape
    
    if P is None:
        P = np.arange(n) # 0 to n-1
        
    P = np.asarray(P, dtype=int)
    
    if S is None:
        S = P.copy()
    else:
        S = np.asarray(S, dtype=int)
        
    if S.size != P.size:
        raise ValueError("S and P must be of the same length.")
        
    f = np.zeros((M, N), dtype=float)
    
    # Iterate through specified planes
    for i in range(len(P)):
        k = P[i] # Index of plane in B to use
        s_val = S[i] # Position index (0 = LSB position -> 2^0)
        
        if k < 0 or k >= n:
            raise ValueError(f"Plane index {k} out of range [0, {n-1}].")
            
        power = 2 ** s_val
        f += B[:, :, k] * power
        
    # Scale to [0, 1]
    # MATLAB: den = 2^n - 1.
    den = 2**n - 1
    f = f / den
    
    return f
