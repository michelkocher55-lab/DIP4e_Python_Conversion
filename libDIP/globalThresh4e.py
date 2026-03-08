import numpy as np

def globalThresh4e(f, delT=0.01):
    """
    Simple global thresholding.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    delT : float, optional
        Convergence tolerance. Defaults to 0.01.
        
    Returns:
    --------
    g : numpy.ndarray (bool)
        Segmented binary image.
    T : float
        Final threshold value.
    k : int
        Number of iterations.
    """
    f = np.array(f, dtype=float)
    
    # Normalization to [0, 1]
    # MATLAB intScaling4e usually scales to [0,1].
    # Here we do it manually if needed, or assume caller handles it?
    # MATLAB code calls intScaling4e(f).
    # If using float image, let's normalize range.
    f_min = f.min()
    f_max = f.max()
    if f_max > f_min:
        f = (f - f_min) / (f_max - f_min)
    
    k = 0
    T = np.mean(f)
    T_old = np.inf
    T_new = T
    
    while abs(T_new - T_old) > delT:
        k += 1
        G = f > T
        
        # Mean of regions
        # Handle empty regions case (rare but possible uniform image)
        m1 = np.mean(f[G]) if np.any(G) else 0.0
        m2 = np.mean(f[~G]) if np.any(~G) else 0.0
        
        T = 0.5 * (m1 + m2)
        T_old = T_new
        T_new = T
        
        # Safety break for oscillation loop (unlikely typically)
        if k > 1000:
            break
            
    # Final segmentation
    g = f > T_new
    
    # Renormalize T to original scale?
    # MATLAB: "The program normalizes the intensity values of F to the range [0,1]."
    # And returns G, T based on that normalized F.
    # So T is in [0, 1].
    
    return g, T_new, k
