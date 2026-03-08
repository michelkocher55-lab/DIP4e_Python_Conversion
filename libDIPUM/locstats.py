
import numpy as np
from skimage.util import img_as_float, img_as_ubyte
from scipy.ndimage import uniform_filter

def locstats(f, m, n, param):
    """
    Performs image enhancement based on local statistics.
    
    Parameters:
    f (ndarray): Input image (assumed uint8).
    m, n (int): Dimensions of the neighborhood (must be odd).
    param (list/tuple): [C, k0, k1, k2, k3].
    
    Returns:
    g (ndarray): Enhanced image (uint8).
    GMF (float): Global Mean of input f.
    GSTDF (float): Global Standard Deviation of input f.
    """
    
    f = np.asarray(f)
    if f.ndim != 2:
        raise ValueError("Input image must be 2D.")
        
    # Preliminaries
    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("The dimensions of the neighborhood must be odd")
        
    # Stats of original image
    GMF = np.mean(f)
    GSTDF = np.std(f)
    
    # Convert to float [0, 1]
    f_norm = img_as_float(f)
    
    # Global stats of normalized image
    GM = np.mean(f_norm)
    GSTD = np.std(f_norm)
    
    C, k0, k1, k2, k3 = param
    
    # Compute local mean
    # uniform_filter computes mean over window.
    # size=(m, n). mode='reflect' is default, MATLAB uses 'symmetric' padding which corresponds to 'reflect' in scipy?
    # scipy 'reflect': d c b a | a b c d | d c b a ("The array is reflected about the edge of the last pixel")
    # scipy 'mirror':  d c b | a b c d | c b a ("The array is reflected about the center of the last pixel")
    # MATLAB 'symmetric': "Pad array with mirror reflections of itself." (Looks like 'reflect' in scipy, checking docs...)
    # MATLAB: 1 2 3 -> 2 1 1 2 3 3 2. (symmetric)
    # Scipy reflect: 1 2 3 -> 1 2 1 2 3 2 1 (Not quite same?)
    # Scipy mirror: 1 2 3 -> 2 1 1 2 3 2.
    # Actually, let's just use 'reflect' which is standard. Differences at boundary are usually minor for this task.
    # MATLAB's 'padarray' symmetric: "replicates the values like a mirror".
    # I will use mode='reflect'.
    
    # uniform_filter uses origin at center by default for odd sizes.
    
    m_local = uniform_filter(f_norm, size=(m, n), mode='reflect')
    
    # Compute local standard deviation
    # std = sqrt(E[x^2] - E[x]^2)
    # E[x^2]:
    m_sq_local = uniform_filter(f_norm**2, size=(m, n), mode='reflect')
    
    # Var = E[x^2] - (E[x])^2
    # Ensure non-negative due to float errors
    var_local = m_sq_local - m_local**2
    var_local[var_local < 0] = 0
    std_local = np.sqrt(var_local)
    
    # Condition mask
    # (mloc >= k0*GM & mloc <= k1*GM) & (stdloc >= k2*GSTD & stdloc <= k3*GSTD)
    
    mask = (m_local >= k0 * GM) & (m_local <= k1 * GM) & \
           (std_local >= k2 * GSTD) & (std_local <= k3 * GSTD)
           
    # Apply enhancement
    # g(idx) = g(idx) * C
    g_out = f_norm.copy()
    g_out[mask] *= C
    
    # Convert back to [0, 255] uint8 using skimage util to handle clipping safe
    # But wait, original code multiplies by 255 then casts to uint8.
    # I should ensure consistency.
    # img_as_ubyte clips and scales properly.
    
    g = img_as_ubyte(np.clip(g_out, 0, 1))
    
    return g, GMF, GSTDF
