import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imageHist4e import imageHist4e
from intScaling4e import intScaling4e

def otsuThresh4e(f):
    """
    Thresholds an image using Otsu's method.
    
    [g, sep, kstar] = otsuThresh4e(f)
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
        
    Returns:
    --------
    g : numpy.ndarray
        Thresholded binary image (bool).
    sep : float
        Separability measure.
    kstar : int
        Optimum threshold.
    """
    
    # Scale to integer [0, 255] range as done in MATLAB
    # MATLAB uses 1-based indexing [1, 256]. Python uses 0-based [0, 255].
    # We'll stick to Python conventions (0-255).
    # intScaling4e usually scales to [0, 1] float or [0, 255] uint8.
    # Let's check intScaling4e usage in MATLAB: intScaling4e(f, 'default', 'integer')
    
    # We can rely on standard numpy ops if intScaling4e is complex, 
    # but the plan said use dependencies.
    # For now, let's coerce to uint8 safely.
    
    if f.dtype != np.uint8:
        # Normalize to 0-255
        f_min = f.min()
        f_max = f.max()
        if f_max > f_min:
            f_scaled = 255 * (f - f_min) / (f_max - f_min)
            f = np.round(f_scaled).astype(np.uint8)
        else:
            f = f.astype(np.uint8)
            
    # Compute normalized histogram
    p = imageHist4e(f) # Should return p for bins 0..255
    if len(p) != 256:
        # imageHist4e might return flexible bins. Ensure 256 for Otsu logic.
        # If imageHist4e is just generic histogram, let's allow it but assume 1D.
        pass
        
    L = len(p)
    # Range of levels k = 0 to L-1
    
    # Cumulative sums P1(k) and P2(k)
    # P1[k] = sum(p[0]...p[k])
    P1 = np.cumsum(p)
    
    # Cumulative means m(k)
    # m[k] = sum(i * p[i] for i in 0..k)
    # Use array indices 0..L-1
    levels = np.arange(L)
    m = np.cumsum(levels * p)
    
    # Global mean mG
    mG = m[-1]
    
    # Between-class variance varB
    # varB = (mG * P1 - m)^2 / (P1 * (1 - P1))
    # Handle division by zero where P1=0 or P1=1
    
    # Only valid range for k is 0 to L-2 usually (since we split into two classes)
    # But vectorized we compute all.
    
    eps = np.finfo(float).eps
    numerator = (mG * P1 - m)**2
    denominator = P1 * (1.0 - P1) + eps
    
    varB = numerator / denominator
    
    # Find optimum threshold kstar
    # kstar is the k that maximizes varB.
    # Note: Thresholding at k means: Classes are [0..k] and [k+1..L-1].
    # So k is the inclusive upper bound of class 1.
    
    maxvarB = np.max(varB)
    # Find all indices where max occurs
    kstar_indices = np.argwhere(varB == maxvarB).flatten()
    
    # Average them and round
    kstar = int(np.round(np.mean(kstar_indices)))
    
    # Global variance
    # varG = sum((i - mG)^2 * p[i])
    varG = np.sum(((levels - mG)**2) * p)
    
    # Separability
    if varG > 0:
        sep = maxvarB / varG
    else:
        sep = 0.0
        
    # Threshold image
    # g = 1 if f > kstar else 0
    g = f > kstar
    
    return g, sep, kstar
