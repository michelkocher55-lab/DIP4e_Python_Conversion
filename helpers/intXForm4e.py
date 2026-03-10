import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from intScaling4e import intScaling4e

def intXForm4e(f, mode, param=None):
    """
    Performs intensity transformations.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input grayscale image (transformed to [0,1]).
    mode : str
        'negative', 'log', 'gamma', 'external'.
    param : float or numpy.ndarray, optional
        Parameter for the transformation.
        - 'gamma': value of gamma. Default 1.0.
        - 'external': numeric array of 256 values (lookup table).
        
    Returns:
    --------
    g : numpy.ndarray
        Transformed image.
    map_func : numpy.ndarray
        The mapping function used (size 256).
    """
    # Defaults
    if param is None:
        if mode == 'gamma':
            param = 1.0
        elif mode == 'external':
            raise ValueError('param must be provided when mode = external')
            
    # Convert input to [0, 1]
    f = intScaling4e(f)
    
    # Create appropriate transformation function, map
    r = np.linspace(0, 1, 256)
    
    if mode == 'negative':
        map_func = 1.0 - r
    elif mode == 'log':
        # Eq. (3-4) with c = 1.0
        map_func = np.log(1.0 + r)
    elif mode == 'gamma':
        # Eq. (3-5) with c = 1.0
        # param is gamma
        # In Python param might be a list if passed as [gamma]. Handle scalar or list-1.
        if np.ndim(param) > 0:
            p = param[0]
        else:
            p = float(param)
        map_func = r ** p
    elif mode == 'external':
        map_func = np.array(param)
        if map_func.size != 256:
            raise ValueError('External param must have 256 values.')
    else:
        raise ValueError("Unrecognized value for mode")
        
    # Perform transformation (Lookup Table)
    # Interp1 equivalent
    # numpy.interp(x, xp, fp)
    # x points to evaluate. f (image).
    # xp x-coordinates of data points (0 to 1).
    # fp y-coordinates of data points (map_func).
    
    x_points = np.linspace(0, 1, len(map_func))
    g = np.interp(f, x_points, map_func)
    
    return g, map_func
