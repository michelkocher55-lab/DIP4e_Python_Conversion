import numpy as np
import collections
from lib.intScaling4e import intScaling4e
from lib.morphoConComp4e import morphoConComp4e
from lib.morphoReconDilate4e import morphoReconDilate4e

def regionGrow4e(f, S, T):
    """
    Perform segmentation by region growing.
    
    R, NR = regionGrow4e(f, S, T)
    
    Parameters
    ----------
    f : numpy.ndarray
        Input image (intensity).
    S : numpy.ndarray or scalar
        Seed mask (array of 0s and 1s) OR scalar seed value.
        If array, 1s indicate seed points.
        If scalar, seeds are all pixels with value S.
    T : numpy.ndarray or scalar
        Threshold.
        
    Returns
    -------
    R : numpy.ndarray
        Labeled regions.
    NR : int
        Number of regions.
    """
    
    # 1. Scale f to integer range [0, 255] (float)
    # Using 'integer' mode suggests 0-255 output.
    # intScaling4e(f, 'default', 'integer') usually returns float in 0-255 range if inputs are 0-1?
    # Or if 'integer' implies uint8? 
    # The MATLAB code uses `double(intScaling4e(..., 'integer'))`.
    # Let's ensure f is float 0-255.
    
    f = intScaling4e(f, 'default', 'integer').astype(float)
    
    # Step 1. Process Seeds
    SI = np.zeros_like(f)
    S1 = []
    
    # Check if S is scalar
    if np.isscalar(S) or (isinstance(S, np.ndarray) and S.size == 1):
        if isinstance(S, np.ndarray):
            S = S.item()
        # S is scalar value
        SI = (f == S).astype(float)
        S1 = [S]
    else:
        # S is array of seeds
        # Eliminate duplicate connected seed locations
        C, NC = morphoConComp4e(S)
        SI = np.zeros_like(S, dtype=float)
        
        # Pick one point per component
        # morphoConComp4e labels components 1..NC
        # To find one index for each, we can use np.where or faster loop
        for i in range(1, NC + 1):
            coords = np.argwhere(C == i)
            if len(coords) > 0:
                # Pick first
                r, c = coords[0]
                SI[r, c] = 1.0
                S1.append(f[r, c])
                
    # Step 2. Form predicate mask fQ
    fQ = np.zeros_like(f, dtype=bool)
    
    # T handling (scalar or array broadcasts)
    
    for seed_val in S1:
        # Predicate: |f - seed| <= T
        diff = np.abs(f - seed_val)
        A = (diff <= T)
        fQ = fQ | A
        
    fQ = fQ.astype(float)
    
    # Step 3. Region Growing via Reconstruction
    # Reconstruct Seeds (SI) into Predicate Mask (fQ).
    # R contains all connected components of fQ that contain at least one seed.
    R_mask, k = morphoReconDilate4e(SI, fQ)
    
    # Step 4. Label connected components
    R, NR = morphoConComp4e(R_mask)
    
    return R, NR
