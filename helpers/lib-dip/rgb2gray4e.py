import numpy as np
from skimage import img_as_float, img_as_ubyte

def rgb2gray4e(f, method='ntsc'):
    """
    Convert RGB image to grayscale.
    
    g = rgb2gray4e(f, method)
    
    Parameters
    ----------
    f : numpy.ndarray
        Input RGB image (H x W x 3). Can be uint8 or float.
    method : str, optional
        Conversion method: 'ntsc' (default) or 'average'.
        'ntsc': Uses weighted sum 0.2989*R + 0.5870*G + 0.1140*B
        'average': Uses simple mean (R+G+B)/3
        
    Returns
    -------
    g : numpy.ndarray
        Grayscale image (uint8).
    """
    
    # 1. Scale to [0, 1] range (handled by img_as_float)
    # in MATLAB code: f = intScaling4e(f) which usually maps min-max or cast.
    # Standard skimage behavior is good enough:
    # uint8 [0,255] -> float [0,1]
    f_double = img_as_float(f)
    
    if f_double.ndim != 3 or f_double.shape[2] != 3:
        raise ValueError("Input must be an RGB image (H x W x 3).")
    
    R = f_double[:, :, 0]
    G = f_double[:, :, 1]
    B = f_double[:, :, 2]
    
    # 2. Convert
    if method == 'average':
        g_double = (R + G + B) / 3.0
    elif method == 'ntsc':
        g_double = 0.2989 * R + 0.5870 * G + 0.1140 * B
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'ntsc' or 'average'.")
        
    # 3. Convert to uint8
    # Code says: floor(g*255) if no toolbox, or im2uint8 (which rounds).
    # We'll use img_as_ubyte which rounds and saturates properly.
    g = img_as_ubyte(g_double)
    
    return g
