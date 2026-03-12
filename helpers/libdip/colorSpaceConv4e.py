import numpy as np
from helpers.libdip.intScaling4e import intScaling4e

def colorSpaceConv4e(f, method):
    """
    Conversion between color spaces.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    method : str
        'rgb2cmy', 'cmy2rgb', 'rgb2cmyk', 'cmyk2rgb'.
        
    Returns:
    --------
    g : numpy.ndarray
        Converted image.
    """
    # Scale f to [0, 1] range using intScaling4e
    # Note: intScaling4e handles uint8 to float conversion.
    f = intScaling4e(f)
    
    method = method.lower()
    
    if method == 'rgb2cmy':
        # RGB to CMY
        # C = 1 - R, M = 1 - G, Y = 1 - B
        # Input assumes 3 channels
        if f.shape[2] != 3:
             raise ValueError("Input for rgb2cmy must have 3 channels.")
             
        R = f[:, :, 0]
        G = f[:, :, 1]
        B = f[:, :, 2]
        
        C = 1.0 - R
        M = 1.0 - G
        Y = 1.0 - B
        
        g = np.stack((C, M, Y), axis=2)
        
    elif method == 'cmy2rgb':
        # CMY to RGB
        # R = 1 - C, etc.
        if f.shape[2] != 3:
             raise ValueError("Input for cmy2rgb must have 3 channels.")
             
        C = f[:, :, 0]
        M = f[:, :, 1]
        Y = f[:, :, 2]
        
        R = 1.0 - C
        G = 1.0 - M
        B = 1.0 - Y
        
        g = np.stack((R, G, B), axis=2)
        
    elif method == 'rgb2cmyk':
        # RGB to CMYK
        if f.shape[2] != 3:
             raise ValueError("Input for rgb2cmyk must have 3 channels.")

        R = f[:, :, 0]
        G = f[:, :, 1]
        B = f[:, :, 2]
        
        C = 1.0 - R
        M = 1.0 - G
        Y = 1.0 - B
        
        # K = min(C, M, Y)
        # np.minimum works element-wise.
        var_K = np.minimum(C, np.minimum(M, Y))
        
        # Handle strict black (K=1) to avoid divide by zero
        # MATLAB: idx = find(var_K == 1);
        # C = (C - var_K) ./ (1 - var_K); ...
        # C(idx) = 0;
        
        # In Python we can use masking or `np.where`.
        
        denom = 1.0 - var_K
        # Avoid division by zero warning
        safe_denom = np.where(denom == 0, 1.0, denom)
        
        C_out = (C - var_K) / safe_denom
        M_out = (M - var_K) / safe_denom
        Y_out = (Y - var_K) / safe_denom
        
        # Where K=1, C,M,Y should be 0.
        black_mask = (var_K == 1.0)
        C_out[black_mask] = 0.0
        M_out[black_mask] = 0.0
        Y_out[black_mask] = 0.0
        
        K_out = var_K
        
        g = np.stack((C_out, M_out, Y_out, K_out), axis=2)
        
    elif method == 'cmyk2rgb':
        # CMYK to RGB
        if f.shape[2] != 4:
             raise ValueError("Input for cmyk2rgb must have 4 channels.")
             
        C = f[:, :, 0]
        M = f[:, :, 1]
        Y = f[:, :, 2]
        K = f[:, :, 3]
        
        R = (1.0 - C) * (1.0 - K)
        G = (1.0 - M) * (1.0 - K)
        B = (1.0 - Y) * (1.0 - K)
        
        g = np.stack((R, G, B), axis=2)
        
    else:
        raise ValueError("Unknown method: " + method)
        
    return g
