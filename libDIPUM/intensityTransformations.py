import numpy as np
from skimage import exposure, util
try:
    from mat2gray import mat2gray
except ImportError:
    # If not found locally, basic fallback or expect it to be in path
    pass 


def intensityTransformations(f, method, *args, **kwargs):
    """
    Grayscale image intensity transformations.
    
    Parameters:
    f: input image
    method: 'neg', 'log', 'gamma', 'stretch', 'specified'
    args: variable arguments depending on method
    
    Returns:
    g: transformed image
    """
    
    # Pre-processing
    # "For 'neg', 'gamma', 'stretch', 'specified': float images outside [0,1] scaled with mat2gray."
    # "For 'log': float images NOT scaled."
    
    f_is_float = f.dtype.kind == 'f'
    original_class = f.dtype
    
    if method != 'log':
        if f_is_float:
            if f.max() > 1 or f.min() < 0:
                f = mat2gray(f)
        else:
            f = util.img_as_float(f)
    else:
        # log method handles float differently logic in MATLAB code:
        # "floating-point images are transformed without being scaled; other images converted using TOFLOAT"
        if not f_is_float:
            f = util.img_as_float(f)
            
    # Dispatch
    if method == 'neg':
        g = 1.0 - f
        
    elif method == 'log':
        # args: C, class_out (optional)
        C = 1.0
        if len(args) >= 1:
            C = args[0]
            
        g = C * np.log(1 + f)
        
        # Handle output class if specified
        if len(args) >= 2:
            cls = args[1]
            if cls == 'uint8':
                g = util.img_as_ubyte(g)
                return g # Return early as we changed class manually
            elif cls == 'uint16':
                g = util.img_as_uint(g)
                return g
                
    elif method == 'gamma':
        # args: gamma
        if len(args) < 1:
            raise ValueError("Method 'gamma' requires gamma value.")
        gam = args[0]
        # imadjust(f, [], [], gamma)
        # skimage doesn't have direct 'adjust_gamma' that does mapping exactly like MATLAB's imadjust with just gamma?
        # Yes, exposure.adjust_gamma(image, gamma=1, gain=1)
        g = exposure.adjust_gamma(f, gamma=gam)
        
    elif method == 'stretch':
        # args: m, E (optional)
        # Defaults: m = mean(f), E = 4.0
        if len(args) >= 1: m = args[0]
        else: m = np.mean(f)
        
        if len(args) >= 2: E = args[1]
        else: E = 4.0
        
        # g = 1./(1 + (m./f).^E)
        # Avoid div by zero if f has 0?
        # MATLAB: (m./f). If f is 0 -> inf. 1/(1+inf) -> 0.
        # Python: 1/(1 + (m/f)**E)
        # Use numpy warnings suppression or careful calc
        with np.errstate(divide='ignore', invalid='ignore'):
             term = (m / (f + 1e-10))**E # Add epsilon to f? Or handle inf.
             # Better: handle term directly.
             # If f=0, term -> inf (if m!=0).
             # den -> inf. g -> 0.
             g = 1.0 / (1.0 + (m / (f + 1e-6))**E)
             
    elif method == 'specified':
        # args: txfun (vector [0,1])
        if len(args) < 1:
             raise ValueError("'specified' requires txfun.")
        txfun = np.asarray(args[0]).flatten()
        if txfun.max() > 1 or txfun.min() < 0:
             raise ValueError("txfun must be in [0, 1].")
             
        # interp1(X, T, f). X is linspace(0, 1, numel(T))
        X = np.linspace(0, 1, len(txfun))
        g = np.interp(f.flatten(), X, txfun).reshape(f.shape)
        
    else:
        raise ValueError(f"Unknown method '{method}'")
        
    # Revert class
    # MATLAB: "revertclass".
    # If input was uint8, revert g (which is float [0,1]) to uint8.
    if original_class == np.uint8:
        g = util.img_as_ubyte(g)
    elif original_class == np.uint16:
        g = util.img_as_uint(g)
    # else float, keep float
    
    return g
