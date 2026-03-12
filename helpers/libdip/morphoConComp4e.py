import numpy as np
from helpers.libdip.morphoReconDilate4e import morphoReconDilate4e

def morphoConComp4e(I):
    """
    Extracts connected components from a binary image using morphological reconstruction.
    
    C, NC = morphoConComp4e(I)
    
    Parameters
    ----------
    I : numpy.ndarray
        Binary image (0/1).
        
    Returns
    -------
    C : numpy.ndarray
        Labelled image (integers).
    NC : int
        Number of components.
        
    Note: This iterative morphological approach can be slow compared to optimized
    labeling algorithms (like skimage.measure.label).
    """
    
    I_work = np.array(I).astype(float) # Working copy
    C = np.zeros_like(I_work)
    NC = 0
    
    # Indices of foreground pixels
    # We can just check np.any(I_work)
    
    while np.any(I_work > 0):
        # Find first foreground pixel
        # argmax finds first True/Max index in flattened array
        flat_idx = np.argmax(I_work > 0)
        
        # Convert to 2D coordinates if needed, or just make Z flat and reshape
        # But morphoReconDilate needs 2D image usually (structuring element).
        Z = np.zeros_like(I_work)
        
        # Unravel index
        # shape is (M, N)
        rows, cols = I_work.shape
        r = flat_idx // cols
        c = flat_idx % cols
        
        Z[r, c] = 1
        
        # Reconstruct component connected to Z, constrained by I_work
        R, _ = morphoReconDilate4e(Z, I_work)
        
        NC += 1
        # Add to C
        C = C + (NC * R)
        
        # Remove from I_work
        I_work = I_work - R
        
        # Ensure non-negative (float precision might drift? usually 0/1 integers)
        I_work[I_work < 0.5] = 0
        
    return C, NC
