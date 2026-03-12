import numpy as np

try:
    # Preferred import when used as part of the `helpers` package.
    from .imnoise2 import imnoise2
except Exception:
    try:
        # Fallback for direct script-style imports.
        from helpers.libdipum.imnoise2 import imnoise2
    except Exception:
        try:
            # Compatibility fallback (if routed through libDIPUM alias package).
            from helpers.libdipum.imnoise2 import imnoise2
        except Exception as e:
            raise ImportError(
                "Could not import imnoise2 required by averaging4noisereduction"
            ) from e

def averaging4noisereduction(f, Q, type_noise, a=None, b=None):
    """
    Generates average of noisy images.
    
    Parameters:
    f: Input image.
    Q: Number of images to average.
    type_noise: specific noise type for imnoise2.
    a, b: Parameters for noise.
    
    Returns:
    A: Averaged image.
    """
    
    # Defaults handling matching MATLAB wrapper logic
    # MATLAB: if nargin == 3 (f, Q, type), a=0, b=1.
    # imnoise2 handling of None usually sets specific defaults per type.
    # But this function sets a global default a=0, b=1 if not provided? 
    # Actually lines 39-42: if nargin == 3 -> a=0, b=1.
    # Note that imnoise2 might have different defaults (e.g. salt & pepper a=0.05).
    # If we pass a=0, b=1 to salt & pepper, we get different behavior.
    # MATLAB code effectively overrides imnoise2 defaults for 3-arg call.
    # We will replicate this behavior if a,b are strictly None.
    
    # Wait, in MATLAB, if you call `averaging4noisereduction(f, 10, 'salt & pepper')`, 
    # it sets a=0, b=1. Then calls `imnoise2(..., 0, 1)`.
    # imnoise2 with a=0, b=1 for salt & pepper:
    #   Ps=0, Pp=1. -> All pixels become pepper (0)? No. 
    #   saltpepper(M,N,a,b) check: if a+b > 1 error. 0+1=1 OK.
    #   Pp=1 means ALL pixels <= 1 are pepper. 
    #   So image becomes black?
    #   That seems like a bad default for S&P.
    #   But that's what the MATLAB code does.
    # We will implement parameter passing as is.
    
    if a is None: a = 0
    if b is None: b = 1
    
    f = np.asarray(f, dtype=np.float32)
    s = f.shape
    
    # Use float for accumulation
    A = np.zeros(s, dtype=np.float32)
    
    for i in range(Q):
        # MATLAB code: A = A + single(f + imnoise2(...))
        # imnoise2 returns clipped fn (usually [0,1]). 
        # With high variance (b=64), fn becomes effectively binary (0 or 1).
        # Adding f again (f + fn) preserves the signal 'f' superimposed on this binary noise.
        # This explains why averaging works despite b=64 (which would wash out f if unclipped).
        
        fn, _ = imnoise2(f, type_noise, a, b)
        A += (f + fn)
        
    A = A / Q
    
    return A
