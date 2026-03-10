import numpy as np

def scanLine4e(f, I, loc):
    """
    Obtains a scan line along a row or column of an image.
    
    s = scanLine4e(f, I, loc)
    
    Parameters
    ----------
    f : numpy.ndarray
        Input image.
    I : int
        Row or column index. (0-based indexing).
    loc : str
        'row' to extract a row, 'col' to extract a column.
        Case-insensitive.
        
    Returns
    -------
    s : numpy.ndarray
        Vector containing the pixel values along the scanline.
    """
    
    loc = loc.lower()
    
    # Validation
    if I < 0:
        raise ValueError("Index I must be non-negative.")
        
    M, N = f.shape[:2] # Handle matches for grayscale or color logic if needed, usually gray
    
    s = None
    
    if loc == 'row':
        if I >= M:
            raise ValueError(f"Index I ({I}) exceeds row dimension ({M}).")
        s = f[I, :]
    elif loc == 'col':
        if I >= N:
            raise ValueError(f"Index I ({I}) exceeds column dimension ({N}).")
        s = f[:, I]
    else:
        raise ValueError("Parameter loc must be 'row' or 'col'.")
        
    return s
