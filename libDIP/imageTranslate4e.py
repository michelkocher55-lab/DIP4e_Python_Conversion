import numpy as np
from skimage.transform import warp, AffineTransform

def imageTranslate4e(f, tx, ty, mode='black'):
    """
    Translates an image spatially.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    tx : float
        Vertical translation (Row shift).
        Positive tx shifts image DOWN.
    ty : float
        Horizontal translation (Column shift).
        Positive ty shifts image RIGHT.
    mode : str, optional
        'black' (default): background is 0.
        'white': background is 1 (or 255 depending on type).
        
    Returns:
    --------
    g : numpy.ndarray
        Translated image. Same size as input.
    """
    # MATLAB Logic:
    # tx is vertical shift (rows).
    # ty is horizontal shift (cols).
    
    # Skimage Affine Transform uses (x, y) = (col, row) convention usually.
    # translation = (tr_x, tr_y) -> (shift_col, shift_row).
    # So tr_x = ty, tr_y = tx.
    
    # We want T s.t. NewCoords = T * OldCoords ??
    # Warp uses inverse map: InputCoords = InverseT * OutputCoords.
    # If we shift right by 10, Output(10) comes from Input(0).
    # Input = Output - 10.
    
    # skimage.transform.AffineTransform(translation=(tx, ty)) defines the FORWARD transform
    # x' = x + tx
    # y' = y + ty
    
    # We pass tform.inverse to warp.
    
    tform = AffineTransform(translation=(ty, tx))
    
    # Determine fill value (cval)
    # skimage warp uses cval argument for background.
    cval = 0
    if mode == 'white':
        if f.dtype == np.uint8:
            cval = 255
        else:
            cval = 1.0 # Assuming float range [0, 1] usually, or max of image?
            # MATLAB intScaling4e usually converts to [0, 1] float.
            # If input is float and max > 1, this might be issue, but standard is 1.0.
            if f.max() > 1.0 and f.dtype != np.uint8:
                cval = f.max()
                
    # Warp
    # preserve_range=True to keep input data range/type value scale (doesn't auto-norm)
    g = warp(f, tform.inverse, output_shape=f.shape, cval=cval, order=0, preserve_range=True)
    
    # Restore original dtype if needed, as warp returns float64
    if f.dtype == np.uint8:
        g = np.round(g).astype(np.uint8)
    
    return g
