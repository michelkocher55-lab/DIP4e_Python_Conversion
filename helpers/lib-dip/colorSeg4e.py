import numpy as np
from lib.imstack2vectors4e import imstack2vectors4e

def colorSeg4e(f, m, T):
    """
    Performs segmentation of a color image.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input RGB image (M, N, 3).
    m : array_like
        A 3-element vector representing the average color (center of sphere).
    T : float
        Threshold for Euclidean distance.
        
    Returns:
    --------
    I : numpy.ndarray
        M-by-N binary image (1 where distance <= T, 0 otherwise).
    """
    
    # Check input dimensions
    f = np.array(f)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("Input image must be RGB.")
        
    M, N, _ = f.shape
    
    # Convert f to vector format
    # imstack2vectors4e returns X (pixels x 3) and R (indices)
    f_vec, _ = imstack2vectors4e(f)
    f_vec = f_vec.astype(float)
    
    # Ensure m is a vector
    m = np.array(m).flatten()
    if m.size != 3:
        raise ValueError("m must be a 3-element vector.")
        
    # Calculate Euclidean distance
    # D = sqrt(sum((f - m)^2, axis=1))
    # f_vec is (K, 3), m is (3,)
    # Broadcasting handles subtraction
    
    D = np.sqrt(np.sum((f_vec - m)**2, axis=1))
    
    # Thresholding
    # J is boolean mask
    J = D <= T
    
    # Create output image (M*N flat first)
    # I = zeros(M*N)
    # I[J] = 1
    # However, if we didn't pass a mask to imstack2vectors4e, 
    # f_vec corresponds exactly to M*N in order.
    # So we can just reshape J.
    
    I_bin = J.astype(int).reshape(M, N)
    
    return I_bin
