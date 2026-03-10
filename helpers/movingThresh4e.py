import numpy as np
import scipy.signal

def movingThresh4e(f, n, c):
    """
    Image segmentation using a moving average threshold.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    n : int
        Number of previous pixels to average (window size).
    c : float
        Threshold multiplier.
        
    Returns:
    --------
    g : numpy.ndarray
        Segmented binary image (bool or 0/1).
    ma : numpy.ndarray
        The moving average image.
    """
    # Scale intensities to [0, 1]
    f = f.astype(float)
    f_min = np.min(f)
    f_max = np.max(f)
    if f_max - f_min > 0:
        f = (f - f_min) / (f_max - f_min)
    else:
        # If constant image, just subtract min (becomes 0)
        f = f - f_min
        
    M, N = f.shape
    
    # Zig-zag scanning to reduce shading bias
    # Flip every other row
    f_zigzag = f.copy()
    f_zigzag[1::2, :] = np.fliplr(f_zigzag[1::2, :])
    
    # Flatten to 1D stream (Row-major)
    f_flat = f_zigzag.flatten()
    
    # Compute Moving Average
    # MATLAB: maf = ones(1,n)/n; ma = filter(maf, 1, f);
    b = np.ones(n) / n
    a = 1
    # lfilter initializes with 0s by default, same as MATLAB filter
    ma_flat = scipy.signal.lfilter(b, a, f_flat)
    
    # Perform Thresholding
    # If value > c * average -> 1 (foreground)
    g_flat = f_flat > (c * ma_flat)
    
    # Reshape back to image
    # Python reshape is Row-major by default, matching the flattening
    g = g_flat.reshape(M, N)
    ma = ma_flat.reshape(M, N)
    
    # Flip alternate rows back
    g[1::2, :] = np.fliplr(g[1::2, :])
    ma[1::2, :] = np.fliplr(ma[1::2, :])
    
    # Convert bool to 0/1 or keep bool? MATLAB returns 0/1 (double or logical).
    # Let's return float 0.0/1.0 to match typical image processing
    g = g.astype(float)
    
    return g, ma
