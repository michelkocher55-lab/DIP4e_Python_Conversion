import numpy as np
from helpers.libdip.intScaling4e import intScaling4e
from helpers.libdip.imPad4e import imPad4e
from helpers.libdip.minusOne4e import minusOne4e

def dftFiltering4e(f, H, padmode='replicate', scaling='no'):
    """
    Filters an image in the frequency domain.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    H : numpy.ndarray
        Filter transfer function. 
        Must be of size 2M-by-2N if padding is used, or M-by-N if padmode is 'none'.
    padmode : str, optional
        'replicate', 'zeros', or 'none'. Default is 'replicate'.
    scaling : str, optional
        'yes' or 'no'. If 'yes', output is scaled to [0, 1]. Default is 'no'.
        
    Returns:
    --------
    g : numpy.ndarray
        Filtered image.
    """
    # Convert image to floating point [0, 1]
    f = intScaling4e(f)
    
    M, N = f.shape[:2]
    
    # Check H size
    if padmode == 'none':
        if H.shape[0] != M or H.shape[1] != N:
            raise ValueError("Filter size must be M-by-N when no padding is used.")
        P, Q = M, N
    else:
        # Default padding is to 2M, 2N
        if H.shape[0] != 2*M or H.shape[1] != 2*N:
            raise ValueError("Filter size must be 2M-by-2N when padding is used.")
        P, Q = 2*M, 2*N
        
    # Pad the image
    if padmode == 'zeros':
        # Pad to P, Q. Since P=2M, Q=2N, we pad by M rows and N cols AFTER.
        # imPad4e(f, R, C, ...) 
        # MATLAB code: imPad4e(f,M,N,'zeros','post') -> adds M lines, N cols. Total 2M, 2N.
        f_padded = imPad4e(f, M, N, padtype='zeros', loc='post')
    elif padmode == 'replicate':
        f_padded = imPad4e(f, M, N, padtype='replicate', loc='post')
    elif padmode == 'none':
        f_padded = f
    else:
        raise ValueError("Unknown padmode. Use 'replicate', 'zeros', or 'none'.")
        
    # Center transform: Multiply by (-1)^(x+y)
    # minusOne4e returns (g, A). We only need g.
    f_centered, _ = minusOne4e(f_padded)
    
    # Compute 2-D FFT
    F = np.fft.fft2(f_centered)
    
    # Multiply by filter transfer function H
    # H is assumed to be centered.
    G = H * F
    
    # Inverse FFT
    g_centered = np.real(np.fft.ifft2(G))
    
    # Undo centering
    g_padded, _ = minusOne4e(g_centered)
    
    # Extract original region (Top-Left quadrant M x N)
    g = g_padded[0:M, 0:N]
    
    # Scaling
    if scaling == 'yes':
        g = intScaling4e(g, mode='full')
        
    return g
