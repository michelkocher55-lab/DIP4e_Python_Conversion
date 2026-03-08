import numpy as np
from scipy.signal import convolve2d
from libDIP.intScaling4e import intScaling4e
from libDIP.imPad4e import imPad4e

def twodConv4e(f, w, param='s'):
    """
    Performs 2D convolution.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    w : numpy.ndarray
        Convolution kernel.
    param : str, optional
        's' for scaling input to [0, 1] (default).
        'ns' for no scaling.

    Returns:
    --------
    g : numpy.ndarray
        Convolved image.
    """
    # Create copies to avoid side effects
    f = np.array(f)
    w = np.array(w).astype(float)
    
    # Pre-processing scaling
    if param == 's':
        f = intScaling4e(f)
    elif param == 'ns':
        f = f.astype(float)
    
    # Kernel dimensions
    WR, WC = w.shape
    rowBorder = int(np.ceil((WR - 1) / 2))
    colBorder = int(np.ceil((WC - 1) / 2))
    
    # Replicate padding to handle boundary effects
    # MATLAB: g = imPad4e(f,rowBorder,colBorder,'replicate','both');
    f_padded = imPad4e(f, rowBorder, colBorder, padtype='replicate', loc='both')
    
    # Convolution
    # MATLAB: g = conv2(f,w,'same');
    # scipy.signal.convolve2d with mode='same'.
    # Note: MATLAB's conv2 rotates kernel by 180 (standard convolution).
    # scipy.signal.convolve2d also does standard convolution (flips kernel).
    # So direct mapping is correct.
    
    # Handle multichannel
    if f.ndim == 3:
        g_padded = np.zeros_like(f_padded)
        for i in range(f.shape[2]):
            g_padded[:,:,i] = convolve2d(f_padded[:,:,i], w, mode='same', boundary='fill', fillvalue=0)
    else:
        g_padded = convolve2d(f_padded, w, mode='same', boundary='fill', fillvalue=0)
        
    # Crop result to remove padding effect
    # MATLAB: g = g(rowBorder + 1:M + rowBorder, colBorder + 1:N + colBorder);
    # In Python 0-indexing:
    # rowBorder : M + rowBorder
    
    # Original M, N
    if f.ndim == 3:
         M, N, _ = f.shape
    else:
         M, N = f.shape
         
    # The padded image has size (M + 2*rowBorder, N + 2*colBorder)
    # conv2 'same' returns same size as input (padded size)
    # We want indices corresponding to original center.
    
    g = g_padded[rowBorder : M + rowBorder, colBorder : N + colBorder]
    
    return g
