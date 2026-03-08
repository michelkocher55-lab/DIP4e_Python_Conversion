import numpy as np
import collections

def cascade4e(g0, g1, num_iter=10):
    """
    Approximates scaling and wavelet functions using the cascade algorithm.
    
    s, w = cascade4e(g0, g1, num_iter=10)
    
    Parameters
    ----------
    g0 : array_like
        Scaling filter coefficients.
    g1 : array_like
        Wavelet filter coefficients.
    num_iter : int
        Number of iterations (default 10).
        
    Returns
    -------
    s : numpy.ndarray
        Approximation of the scaling function (phi).
    w : numpy.ndarray
        Approximation of the wavelet function (psi).
    """
    
    g0 = np.array(g0).flatten()
    g1 = np.array(g1).flatten()
    
    # Scale filters by sqrt(2) as per MATLAB code
    g0 = np.sqrt(2) * g0
    g1 = np.sqrt(2) * g1
    
    # Initial approximations
    s = g0.copy()
    w = g1.copy()
    
    # MATLAB loop does upsampling, then convolution with g0.
    # Logic:
    # x = upsample(s)
    # s_new = conv(x, g0)
    # x2 = upsample(w)
    # w_new = conv(x2, g0)
    
    for i in range(num_iter):
        # Upsample s
        # MATLAB: x(1:2:length(s)*2) = s; x(2:2:end) = 0;
        # Python: 
        x = np.zeros(2 * len(s))
        x[0::2] = s
        
        # Upsample w
        x2 = np.zeros(2 * len(w))
        x2[0::2] = w
        
        # Convolve
        s = np.convolve(x, g0, mode='full')
        w = np.convolve(x2, g0, mode='full') # Note: w also convolved with g0!
        
    return s, w
