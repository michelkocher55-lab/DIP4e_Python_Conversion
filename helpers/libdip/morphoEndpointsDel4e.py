import numpy as np
from helpers.libdip.morphoThin4e import morphoThin4e
from helpers.libdip.morphoMatch4e import morphoMatch4e

def morphoEndpointsDel4e(Ithin, numiter=None, mode='nosingletons'):
    """
    Deletes end points from thinned image.
    
    Idel = morphoEndpointsDel4e(Ithin, numiter, mode='nosingletons')
    
    Parameters
    ----------
    Ithin : numpy.ndarray
        Thinned binary image.
    numiter : int, optional
        Number of iterations (pixels to shave off).
        Defaults to Inf (remove all branches until only closed loops or single points remain?).
        Wait, MATLAB code defaults to passed numiter.
        Wait, there is no default for numiter in MATLAB checks (Lines 28-30 only set mode).
        If numiter is not passed, nargin check fails?
        Actually MATLAB code says "if nargin == 2, mode=...". It assumes numiter is passed.
        If user omits it, it errors?
        I will make it optional for flexibility, defaulting to 1?
        Or Inf? Usually we want to prune N pixels.
        Let's require it or default to something safe. 
        Usually pruning is controlled.
    mode : str
        'nosingletons' (default) or 'leave'.
        
    Returns
    -------
    Idel : numpy.ndarray
        Result image.
    """
    
    if numiter is None:
        # Default behaviour if not specified?
        # Let's default to 1 iteration if not specified to be safe, or just follow caller.
        numiter = 1 
        
    # Construct SEs for Pruning (Fig 9.26b)
    # 8 SEs
    B_stack = np.zeros((3, 3, 8))
    
    # B1:
    # 2 0 0
    # 1 1 0
    # 2 0 0
    b1 = np.array([
        [2, 0, 0],
        [1, 1, 0],
        [2, 0, 0]
    ])
    
    # B5: (Line 22)
    # 1 0 0
    # 0 1 0
    # 0 0 0
    b5 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    # Rotate assignments
    # Stack structure:
    # 1: b1
    # 2: rot(b1)
    # ...
    # 5: b5
    # 6: rot(b5)
    # ...
    
    # Actually code assigns:
    # B(:,:,1) = b1
    # B(:,:,2) = rot(b1, -1) ...
    # B(:,:,5) = b5
    # ...
    
    curr_b1 = b1
    for k in range(0, 4):
        B_stack[:, :, k] = curr_b1
        curr_b1 = np.rot90(curr_b1, 3) # CW
        
    curr_b5 = b5
    for k in range(4, 8):
        B_stack[:, :, k] = curr_b5
        curr_b5 = np.rot90(curr_b5, 3) # CW
        
    # Thinning
    Idel = morphoThin4e(Ithin, B_stack, numiter)
    
    # Check mode
    if mode == 'nosingletons':
        # Remove isolated single points
        # Bsingle:
        # 0 0 0
        # 0 1 0
        # 0 0 0
        Bsingle = np.zeros((3, 3))
        Bsingle[1, 1] = 1
        
        # Find matches (Perfect match means surrounded by 0s)
        # Note: morphoMatch4e uses 'same' mode by default.
        # But we must ensure padval=0 (background). Default is 0.
        singlePoints = morphoMatch4e(Idel, Bsingle, padval=0)
        
        # Eliminate
        # Idel[singlePoints == 1] = 0
        Idel = Idel * (1 - (singlePoints == 1.0))
        
    return Idel
