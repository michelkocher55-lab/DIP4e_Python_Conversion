
import numpy as np
from scipy.ndimage import correlate
from skimage.util import img_as_float

def fillgaps(B, L):
    """
    Fills gaps of L or fewer 0s in each row of B.
    B must be a binary image (0s and 1s).
    """
    G = B.copy()
    rows, cols = G.shape
    
    # Iterate over each row
    for r in range(rows):
        row_data = G[r, :]
        # Find indices of non-zero elements
        indices = np.nonzero(row_data)[0]
        
        if len(indices) < 2:
            continue
            
        # Check gaps between consecutive 1s
        # indices[i] is the start 1, indices[i+1] is the next 1
        # gap is indices[i+1] - indices[i] - 1
        
        # Vectorized check for the row? Or simple loop.
        # Simple loop is fine per row.
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i+1]
            gap_size = end - start - 1
            
            if 0 < gap_size <= L:
                G[r, start+1:end] = 1
                
    return G

def edgelinklocal(f, P, A, TA, L):
    """
    Links edge points based on local analysis.
    
    Parameters:
        f: Input image.
        P: Magnitude threshold factor (0 < P < 1). 
           Threshold E = P * max(Gradient Magnitude).
           Keep pixel if Mag > E.
        A: Target Angle in degrees.
        TA: Angle tolerance in degrees.
            Keep pixel if Angle is in [A - TA, A + TA].
        L: Max gap length to fill (in horizontal direction).
        
    Returns:
        G: Binary linked edge image.
        MAG: Gradient magnitude.
        ALPHA: Gradient angle (degrees).
        Gx: Gradient x-component.
        Gy: Gradient y-component.
    """
    
    f = img_as_float(f)
    
    # Standard Sobel masks from the script
    # sx = [-1 -2 -1; 0 0 0; 1 2 1];
    # sy = sx';
    sx = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)
    sy = sx.T
    
    # MATLAB: imfilter(f, sx, 'replicate') -> Equivalent to correlation with edge padding (nearest or reflection)
    # Replicate padding in MATLAB repeats the border value. Scipy 'nearest' does the same.
    # imfilter performs correlation by default.
    
    Gx = correlate(f, sx, mode='nearest')
    Gy = correlate(f, sy, mode='nearest')
    
    # Magnitude
    MAG = np.sqrt(Gx**2 + Gy**2)
    E = P * MAG.max()
    
    # Angle
    # MATLAB form is atan(Gx./(Gy + eps))*180/pi.
    # arctan2 gives the same orientation intent without divide-by-zero warnings.
    ALPHA = np.degrees(np.arctan2(Gx, Gy))
    
    # Select Pixels
    # I = find(MAG>E & abs(ALPHA)>A-TA & abs(ALPHA)<A+TA);
    # Note: MATLAB code used abs(ALPHA) because "an edge at -90 deg is the same edge as +90 deg".
    # And checks if abs(ALPHA) is within A +/- TA.
    # The condition in the file: abs(ALPHA)>A-TA & abs(ALPHA)<A+TA
    
    # Let's match strictly.
    abs_alpha = np.abs(ALPHA)
    
    mask = (MAG > E) & (abs_alpha > (A - TA)) & (abs_alpha < (A + TA))
    
    G = np.zeros_like(f, dtype=int)
    G[mask] = 1
    
    # Fill Gaps
    G = fillgaps(G, L)
    
    return G, MAG, ALPHA, Gx, Gy
