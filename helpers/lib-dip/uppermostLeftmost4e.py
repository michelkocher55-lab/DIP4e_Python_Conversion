import numpy as np

def uppermostLeftmost4e(b):
    """
    Finds the uppermost-leftmost point of a closed boundary.
    
    ulp = uppermostLeftmost4e(b)
    
    Parameters
    ----------
    b : numpy.ndarray
        (N, 2) array of boundary coordinates [row, col].
        
    Returns
    -------
    ulp : numpy.ndarray
        (1, 2) array [row, col] of the uppermost-leftmost point.
    """
    
    b = np.array(b)
    
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("Input b must be an N-by-2 array.")
        
    rows = b[:, 0]
    cols = b[:, 1]
    
    # 1. Find Uppermost (Minimum Row index)
    min_row = np.min(rows)
    
    # 2. Identify candidates (all points with min_row)
    # In Python, we can just mask or use argwhere, assuming numerical exactness for integer coordinates.
    # If coordinates are float, using a tolerance might be safer, but DIP boundaries are usually pixel indices.
    mask_rows = (rows == min_row)
    
    # 3. Find Leftmost (Minimum Col index) among candidates
    candidate_cols = cols[mask_rows]
    min_col_at_min_row = np.min(candidate_cols)
    
    ulp = np.array([min_row, min_col_at_min_row])
    
    # Note: MATLAB returns 1-by-2. Numpy array [r, c] is shape (2,). 
    # If strictly 1-by-2 desired:
    ulp = ulp.reshape(1, 2)
    
    return ulp
