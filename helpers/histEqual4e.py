import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from intScaling4e import intScaling4e
from imageHist4e import imageHist4e
from intXForm4e import intXForm4e

def histEqual4e(f):
    """
    Histogram equalization.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
        
    Returns:
    --------
    g : numpy.ndarray
        Equalized image with intensities in [0, 1].
    """
    # Scale to [0, 1]
    f = intScaling4e(f)
    
    # Compute normalized histogram
    h = imageHist4e(f)
    
    # Compute CDF
    # Use numpy.cumsum
    cdf = np.cumsum(h)
    
    # Use intXForm4e to map intensities
    # mode='external', param=cdf
    g, _ = intXForm4e(f, 'external', cdf)
    
    return g
