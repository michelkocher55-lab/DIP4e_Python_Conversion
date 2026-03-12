import numpy as np
import warnings
from helpers.libdip.morphoHitMiss4e import morphoHitMiss4e

def morphoThin4e(I, B=None, numiter=None):
    """
    Morphological thinning of binary image.
    
    T = morphoThin4e(I, B=None, numiter=None)
    
    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray, optional
        Structuring elements (stack of K elements).
        If None, uses default thinning SEs (Homotopic Thinning).
        Shape: (m, n, K).
    numiter : int, optional
        Number of iterations. Default Inf (until stability).
        
    Returns
    -------
    T : numpy.ndarray
        Thinned image.
    """
    
    I = (I > 0).astype(float) # Ensure binary
    
    # Defaults
    if numiter is None:
        numiter = float('inf')
        
    if B is None:
        # Default Thinning SEs (Fig 9.23 DIP4E)
        # 3x3x8
        B_stack = np.zeros((3, 3, 8))
        
        # B1:
        # 0 0 0
        # 2 1 2
        # 1 1 1
        b1 = np.array([
            [0, 0, 0],
            [2, 1, 2],
            [1, 1, 1]
        ])
        
        # B2:
        # 2 0 0
        # 1 1 0
        # 1 1 2
        b2 = np.array([
            [2, 0, 0],
            [1, 1, 0],
            [1, 1, 2]
        ])
        
        B_stack[:,:,0] = b1
        B_stack[:,:,1] = b2
        
        # Others are rotations of B1, B2
        # B3 = rot90(B1, -1) (CW)
        # B4 = rot90(B2, -1)
        # etc.
        
        curr_b1 = b1
        curr_b2 = b2
        
        for k in range(2, 8, 2):
            # Rotate CW matches MATLAB rot90(A, -1)
            # numpy rot90(A, k=1) is CCW. So k=-1 is CW. (or k=3).
            curr_b1 = np.rot90(curr_b1, 3) 
            curr_b2 = np.rot90(curr_b2, 3)
            B_stack[:,:,k] = curr_b1
            B_stack[:,:,k+1] = curr_b2
            
        B = B_stack
    else:
        B = np.array(B)
        # Ensure B is (m,n,K). If (m,n), expand dims
        if B.ndim == 2:
            B = B[:, :, np.newaxis]
            
    K = B.shape[2]
    
    currentH = I.copy()
    stopCondition = False
    iterCount = 0
    
    while not stopCondition:
        params_changed_count = 0 # Count how many SEs produced NO change
        
        # Per iteration: cycle through all K SEs
        for k in range(K):
            currentB = B[:, :, k]
            
            # HM
            hm = morphoHitMiss4e(currentH, currentB)
            
            # Thin: H & ~HM
            # Remove pixels that matched
            # matched pixels are candidates for deletion
            
            # T = currentH & ~hm
            T = currentH * (1 - hm) # float multiplication
            
            if np.array_equal(currentH, T):
                params_changed_count += 1
            
            currentH = T
            
        iterCount += 1
        
        # Stop if no changes in any sub-iteration OR max iter reached
        if params_changed_count == K or iterCount >= numiter:
            stopCondition = True
            
    return currentH
