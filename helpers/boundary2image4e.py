import numpy as np
from skimage.draw import polygon

def boundary2image4e(b, M, N, fill=0):
    """
    Embeds a boundary into an image.
    
    Parameters:
    -----------
    b : numpy.ndarray
        (K, 2) array containing the integer coordinates of a boundary.
        Format: [row, col] (0-based indexing).
    M : int
        Number of rows in the output image.
    N : int
        Number of columns in the output image.
    fill : int, optional
        If 1, the inside of the boundary is filled with 1s.
        If 0 (default), only the boundary is displayed.
        
    Returns:
    --------
    image : numpy.ndarray
        M-by-N binary image (0s and 1s).
    """
    
    # Check input b
    b = np.array(b)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError('The boundary must be of size K-by-2')
        
    # Make sure coordinates are integers
    b = np.round(b).astype(int)
    
    rows = b[:, 0]
    cols = b[:, 1]
    
    # Initialize image
    image = np.zeros((M, N), dtype=int)
    
    # Filter valid coordinates to prevent index errors
    valid_mask = (rows >= 0) & (rows < M) & (cols >= 0) & (cols < N)
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    
    if len(valid_rows) == 0 and len(b) > 0:
        # Warning: all boundary points are outside the image?
        pass

    if fill == 1:
        # Fill the polygon
        # skimage.draw.polygon fills the polygon defined by vertices (r, c)
        # Note: if b defines a dense boundary (all pixels), polygon still works.
        rr, cc = polygon(rows, cols, shape=(M, N))
        image[rr, cc] = 1
    else:
        # Boundary only
        if len(valid_rows) > 0:
            image[valid_rows, valid_cols] = 1
            
    return image
