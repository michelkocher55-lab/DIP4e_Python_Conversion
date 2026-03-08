
import numpy as np

def iseven(n):
    return n % 2 == 0

def logdogfilter(n, sigma2, k, option, norm=0):
    """
    LOGDOGFILTER Laplacian of Gaussian and difference of gaussians filters.
    
    [LOG, DOG, PL, PD] = LOGDOGFILTER(N, SIGMA2, K, OPTION, NORM)
    Generates a Laplacian of a Gaussian (LOG) and a difference of Gaussians
    (DOG) filter, both of size N-by-N.
    
    Parameters:
        n: Filter size. If even, incremented by 1.
        sigma2: Standard deviation of smaller Gaussian in DOG.
        k: Ratio sigma1 = k*sigma2 (k > 1).
        option: 'auto' or value for sigma of LoG filter.
        norm: Normalization value for origin. If 0, no normalization.
        
    Returns:
        LOG: LoG filter.
        DOG: DoG filter.
        PL: Horizontal profile of LoG through center.
        PD: Horizontal profile of DoG through center.
    """
    
    if k <= 1:
        raise ValueError('sigma1 must be greater than sigma2')
        
    sigma1 = k * sigma2
    s1 = sigma1**2
    s2 = sigma2**2
    
    if iseven(n):
        n = n + 1
        
    if isinstance(option, str) and option == 'auto':
        sigma = np.sqrt( (s1*s2 / (s1 - s2)) * np.log(s1/s2) )
    else:
        sigma = float(option)
        
    # Generate meshgrid
    width = (n - 1) // 2
    x = np.arange(-width, width + 1)
    y = np.arange(-width, width + 1)
    X, Y = np.meshgrid(x, y)
    
    # Generate DOG filter
    eps = np.finfo(float).eps
    S1 = 2*s1 + eps
    S2 = 2*s2 + eps
    k1 = 1 / (np.pi * S1)
    k2 = 1 / (np.pi * S2)
    num = X**2 + Y**2
    G1 = k1 * np.exp(-num / S1)
    G2 = k2 * np.exp(-num / S2)
    DOG = G1 - G2
    
    # Generate LOG filter
    S = 2 * (sigma**2) + eps
    K = 1 / (np.pi * S)
    # LOG = k*((num - s)./(sigma^4 + eps)).*exp(-num/s);
    # Note: MATLAB code used k (small k) which overwritten input argument k? 
    # Let's check MATLAB source again.
    # L1: function ... k ...
    # L70: k = 1/(pi*s); 
    # Yes, it overwrites.
    
    sig4 = sigma**4 + eps
    LOG = K * ((num - S) / sig4) * np.exp(-num / S)
    
    # Normalization
    if norm != 0:
        # Scale so value at origin is equal to NORM.
        # Value at origin is the min value (negative peak for inverted LoG).
        # MATLAB: LOG = (LOG./abs(min(LOG(:))))*norm;
        # Since LoG at origin is negative.
        LOG = (LOG / np.abs(LOG.min())) * norm
        DOG = (DOG / np.abs(DOG.min())) * norm
        
    # Profiles
    c = int((n - 1) // 2)
    # MATLAB: PL = LOG(1:end, c); where c is center index (1-based).
    # Python 0-based center index is width.
    # MATLAB 1-based: (n-1)/2 + 1. e.g. n=5 => 3.
    # Python 0-based: 2.
    
    PL = LOG[:, c]
    PD = DOG[:, c]
    
    return LOG, DOG, PL, PD
