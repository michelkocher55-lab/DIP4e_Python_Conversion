import numpy as np
import warnings
from lib.morphoThin4e import morphoThin4e
from lib.morphoHitMiss4e import morphoHitMiss4e
from lib.morphoDilate4e import morphoDilate4e

def morphoPrun4e(I, numthin, numdil):
    """
    Morphological pruning of the foreground of binary image I.
    
    P = morphoPrun4e(I, numthin, numdil)
    
    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    numthin : int
        Number of thinning iterations (to remove spurs).
    numdil : int
        Number of dilation iterations (to restore main tips).
        Generally numdil <= numthin.
        
    Returns
    -------
    P : numpy.ndarray
        Pruned image.
    """
    
    if numdil > numthin:
        warnings.warn("Warning: Are you sure you want numdil > numthin?")
        
    I = (I > 0).astype(float)
    
    # Define Structuring Elements (Endpoint Detectors)
    # 3x3x8 stack
    B_stack = np.zeros((3, 3, 8))
    
    # Sequence 1-4
    # B1: [2 0 0; 1 1 0; 2 0 0]
    # Note: 2 is Don't Care.
    b1 = np.array([
        [2, 0, 0],
        [1, 1, 0],
        [2, 0, 0]
    ])
    
    # Sequence 5-8
    # B5: [1 0 0; 0 1 0; 0 0 0]
    b5 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    # Fill stack with rotations (CW matches MATLAB rot90 -1)
    # rot90(m, -1) in Matlab is CW.
    # np.rot90(m, 3) is CW (since default k=1 is CCW).
    
    curr_b1 = b1
    curr_b5 = b5
    
    for k in range(4):
        B_stack[:, :, k] = curr_b1
        B_stack[:, :, k+4] = curr_b5
        
        # Rotate for next
        curr_b1 = np.rot90(curr_b1, 3)
        curr_b5 = np.rot90(curr_b5, 3)
        
    # 1. Thinning
    # Remove endpoints numthin times.
    X1 = morphoThin4e(I, B_stack, numthin)
    
    # 2. Find Endpoints
    # Union of HitMiss with all SEs.
    X2 = np.zeros_like(X1)
    K = 8
    for k in range(K):
        currentB = B_stack[:, :, k]
        endPoints = morphoHitMiss4e(X1, currentB)
        # X2 = X2 | endPoints
        X2 = np.maximum(X2, endPoints)
        
    # 3. Dilate endpoints constrained by I (Reconstruct tips)
    B3 = np.ones((3, 3))
    currentX3 = X2.copy()
    
    for i in range(numdil):
        dilated = morphoDilate4e(currentX3, B3)
        # Constrain by I
        # X3 = dilated & I
        currentX3 = dilated * I
        
    X3 = currentX3
    
    # 4. Result P = X1 | X3
    P = np.maximum(X1, X3)
    
    return P
