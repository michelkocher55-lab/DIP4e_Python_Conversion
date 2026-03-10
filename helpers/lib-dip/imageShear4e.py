import numpy as np
from skimage.transform import warp, AffineTransform
from skimage import img_as_float, img_as_ubyte

def imageShear4e(f, sv=0, sh=0, method='nearest'):
    """
    Shears an image vertically and/or horizontally.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    sv : float
        Vertical shear factor.
    sh : float
        Horizontal shear factor.
    method : str, optional
        Interpolation method: 'nearest' (default, matches MATLAB) or 'linear'.
        
    Returns:
    --------
    g : numpy.ndarray
        Sheared image. Same size as input, zero-padded.
    """
    # Create the shear transformation matrix
    # Based on MATLAB derivation: S = [(1+sv*sh) sv 0; sh 1 0; 0 0 1] on [row; col; 1]
    # In Python (x, y) = (col, row). 
    # x' = 1*x + sh*y
    # y' = sv*x + (1+sv*sh)*y
    
    # Skimage AffineTransform matrix is 3x3:
    # [[a0, a1, a2],
    #  [b0, b1, b2],
    #  [0,  0,  1 ]]
    # x' = a0*x + a1*y + a2
    # y' = b0*x + b1*y + b2
    
    matrix = np.array([
        [1, sh, 0],
        [sv, 1 + sv * sh, 0],
        [0, 0, 1]
    ])
    
    # Define Forward Transform
    tform = AffineTransform(matrix=matrix)
    
    # Warp expects the INVERSE map (Output -> Input).
    # tform.inverse gives us that.
    
    # Order: 0 = Nearest Neighbor, 1 = Linear
    order = 0 if method == 'nearest' else 1
    
    # Warp
    # preserve_range=True keeps range (important if float input)
    g = warp(f, tform.inverse, output_shape=f.shape, order=order, preserve_range=True)
    
    # Cast back to original type if needed
    if f.dtype == np.uint8:
        g = np.round(g).astype(np.uint8)
        
    return g
