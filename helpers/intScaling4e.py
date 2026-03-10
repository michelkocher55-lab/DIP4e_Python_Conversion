
import numpy as np
from skimage.util import img_as_ubyte

def intScaling4e(f, mode='default', type_out='floating'):
    """
    Scales the intensity values of an input image.
    
    Parameters:
    f: input image (ndarray)
    mode: 'default' or 'full'
    type_out: 'floating' or 'integer'
    
    Returns:
    g: scaled image
    """
    
    # Handle inputs if not specified (Python default args handle basic case)
    # Check if input is uint8
    is_uint8 = f.dtype == np.uint8
    
    # Check mix/max
    f_min = np.min(f)
    f_max = np.max(f)
    
    # Automatic mode selection logic from MATLAB:
    # If floating point and outside [0, 1], force mode='full', type='floating'
    # unless mode was explicitly provided? MATLAB script says:
    # "If F is floating point with values outside the range [0,1], then MODE = 'full' and TYPE = 'floating' are forced."
    # But only if arguments 2 and 3 are NOT provided.
    # Here we have default arguments. If user calls intScaling4e(f), mode is 'default'.
    # We need to detect if user *provided* mode. In Python, can checks args or just rely on explicit defaults.
    # Let's check logic: "If mode is 'default'..."
    
    # Since we can't easily detect "not provided" vs "default" without using None or **kwargs,
    # let's assume 'default' means "apply default logic including auto-detection".
    
    computed_mode = mode
    computed_type = type_out
    
    if mode == 'default':
        if not is_uint8:
            if f_min < 0 or f_max > 1:
                computed_mode = 'full'
                computed_type = 'floating'
            else:
                 pass # default logic applies
    
    # Convert to float
    g = f.astype(float)
    g_max_all = np.max(g)
    
    if computed_mode == 'default':
        if g_max_all > 255: # Logic from MATLAB: if max > 255, g = g/max
             g = g / g_max_all
        elif is_uint8: # If original was uint8
             g = g / 255.0
        else:
             pass # Already float <= 1 (presumably)
             
    elif computed_mode == 'full':
        g = g - np.min(g)
        max_val = np.max(g)
        if max_val != 0:
            g = g / max_val
            
    # Output type
    if computed_type == 'integer':
        g = img_as_ubyte(g)
        
    return g
