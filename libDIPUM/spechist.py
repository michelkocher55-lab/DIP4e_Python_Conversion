
import numpy as np
from scipy.signal import convolve2d

def spechist(f, h):
    """
    Performs exact histogram specification.
    
    Parameters:
    f (ndarray): Input image.
    h (ndarray): Specified histogram (unnormalized). 
                 Sum of h should equal number of pixels in f.
                 
    Returns:
    g (ndarray): Enhanced image (uint8).
    
    References:
    Based on 'spechist.m' from DIPUM4e (Coutu method).
    """
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=int).flatten()
    
    if np.sum(h) != f.size:
        # Try to normalize/scale h to match image size?
        # MATLAB code says "must be unnormalized... components must equal number of pixels".
        # We will follow this strict requirement or raise warning?
        # Let's adjust h to match sum if close? Or just error.
        # But for robustness let's renormalize h to sum to f.size.
        # However, to be strict with the algorithm which iterates exactly:
        if np.sum(h) != f.size:
             raise ValueError(f"Sum of histogram ({np.sum(h)}) must match image size ({f.size}).")
             
    # Kernels
    f1 = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]]) / 4.0
                   
    f2 = np.array([[1, 0, 1],
                   [0, 0, 0],
                   [1, 0, 1]]) / 4.0
                   
    f3 = np.array([[0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0]]) / 4.0
                   
    f4 = np.array([[0, 1, 0, 1, 0],
                   [1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1],
                   [0, 1, 0, 1, 0]]) / 8.0
                   
    f5 = np.array([[1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1]]) / 4.0
    
    # Convolution parameters
    # MATLAB conv2(f, k, 'same') uses zero padding by default.
    mode = 'same'
    boundary = 'fill'
    fillvalue = 0
    
    c1 = convolve2d(f, f1, mode=mode, boundary=boundary, fillvalue=fillvalue)
    c2 = convolve2d(f, f2, mode=mode, boundary=boundary, fillvalue=fillvalue)
    c3 = convolve2d(f, f3, mode=mode, boundary=boundary, fillvalue=fillvalue)
    c4 = convolve2d(f, f4, mode=mode, boundary=boundary, fillvalue=fillvalue)
    c5 = convolve2d(f, f5, mode=mode, boundary=boundary, fillvalue=fillvalue)
    
    L = 256.0
    
    # Construct priority image
    # Note: Coefficients are large, check for overflow? Python float64 is fine.
    imag = (f * L**5 + 
            c1 * L**4 + 
            c2 * L**3 + 
            c3 * L**2 + 
            c4 * L + 
            c5)
            
    # Reshape and Sort
    # MATLAB: sir = reshape(imag, 1, M*N). Column-major.
    # [ord, ind] = sort(sir).
    # Python default is Row-major.
    # To match MATLAB exactly, we should flatten in Fortran order 'F'.
    flat_imag = imag.flatten(order='F')
    
    # Argsort
    # MATLAB sort is stable? "If the array contains repeated elements, sort preserves their relative order." (Stable sort).
    # NumPy argsort uses quicksort by default (unstable). Use 'stable' (mergesort).
    ind = np.argsort(flat_imag, kind='stable')
    
    # Assign values
    g_flat = np.zeros_like(flat_imag, dtype=np.uint8)
    
    # Assignment loop optimization
    # Instead of while loops, we can create a lookup array or expand h.
    # target_vals = [0]*h[0] + [1]*h[1] ...
    # This matches the logic: for each pixel in sorted order, assign next available histogram value.
    
    # Create array of target values
    # np.repeat([0, 1, 2...], h)
    levels = np.arange(len(h))
    target_vals = np.repeat(levels, h)
    
    # Verify size
    if target_vals.size != ind.size:
         # Should not happen if check above passed
         # Truncate or pad?
         pass
         
    # Assign
    # The pixel at ind[k] gets target_vals[k]
    # In MATLAB:
    # for pix=1:N*M
    #    x = ind(pix) ...
    #    g(x) = current_level
    #    decrement h count...
    
    # Vectorized assignment:
    g_flat[ind] = target_vals
    
    # Reshape back to image
    # We flattened with 'F' (column major), so reshape with 'F'.
    g = g_flat.reshape(f.shape, order='F')
    
    return g
