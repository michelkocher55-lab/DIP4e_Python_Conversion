
import numpy as np

def boundarydir(x, y=None, orderout=None):
    """
    Determine the direction of a sequence of planar points.
    
    Parameters:
    x (array-like): x coordinates or Nx2 array of (x, y) coordinates.
    y (array-like, optional): y coordinates if x is 1D.
    orderout (str, optional): Desired output order ('cw' or 'ccw').
                              If provided, returns (dir, x_out, y_out).
                              
    Returns:
    dir (str): 'cw' or 'ccw'.
    x_out (array): Ordered x coordinates (only if orderout is specified).
    y_out (array): Ordered y coordinates (only if orderout is specified).
    """
    
    # Input handling
    x = np.array(x)
    if y is None:
        if x.ndim == 2 and x.shape[1] == 2:
            y = x[:, 1]
            x = x[:, 0]
        else:
            raise ValueError("Invalid input arguments. Provide x and y, or x as Nx2 array.")
    else:
        y = np.array(y)
        
    x = x.flatten()
    y = y.flatten()
    
    # Handle duplicate start/end
    restore_end = False
    if len(x) > 1 and x[0] == x[-1] and y[0] == y[-1]:
        x_clean = x[:-1]
        y_clean = y[:-1]
        restore_end = True
    else:
        x_clean = x
        y_clean = y
        
    # Check for other duplicates (not allowed per MATLAB spec)
    # Combining to check uniqueness
    points = np.column_stack((x_clean, y_clean))
    unique_points = np.unique(points, axis=0)
    if len(points) != len(unique_points):
        # Allow robustness or raise error? MATLAB raises error.
        # "No duplicate points except first and last are allowed."
        raise ValueError('No duplicate points except first and last are allowed.')

    # Find topmost, leftmost point (min x, then min y)
    # Note: MATLAB `find(x0 == min(x0))` finds indices.
    # We want index of min x. If ties, min y.
    
    min_x = np.min(x_clean)
    indices_min_x = np.where(x_clean == min_x)[0]
    
    if len(indices_min_x) == 1:
        idx = indices_min_x[0]
    else:
        # Ties in x, minimize y
        sub_y = y_clean[indices_min_x]
        min_y = np.min(sub_y)
        # Find index in sub_y
        idx_in_sub = np.where(sub_y == min_y)[0][0]
        idx = indices_min_x[idx_in_sub]
        
    # Scroll data so (x1, y1) is first
    # MATLAB: circshift(x0, [-(I - 1), 0]) shifts UP by I-1.
    # Python roll shifts by +k to right. To shift k to left, roll by -k.
    # We want index `idx` to become 0.
    x_shifted = np.roll(x_clean, -idx)
    y_shifted = np.roll(y_clean, -idx)
    
    # Check direction using 3 points: last, first, second (of shifted)
    # A = [x_prev, y_prev, 1; x_curr, y_curr, 1; x_next, y_next, 1]
    # Matrix rows:
    # 1. End of shifted (prev)
    # 2. Start of shifted (curr)
    # 3. Second of shifted (next)
    
    x_prev = x_shifted[-1]
    y_prev = y_shifted[-1]
    
    x_curr = x_shifted[0]
    y_curr = y_shifted[0]
    
    x_next = x_shifted[1]
    y_next = y_shifted[1]
    
    A = np.array([
        [x_prev, y_prev, 1],
        [x_curr, y_curr, 1],
        [x_next, y_next, 1]
    ])
    
    determinant = np.linalg.det(A)
    
    direction = 'cw'
    if determinant > 0:
        direction = 'ccw'
        
    if orderout is None:
        return direction
    else:
        x_out = x_clean
        y_out = y_clean
        
        # If mismatch, reverse
        if direction != orderout:
            # Keep first point, reverse rest?
            # MATLAB: x0(2:end) = flipud(x0(2:end));
            # This keeps the guaranteed convex point at start, but reverses traversal.
            # Wait, `x0` in MATLAB was shifted! 
            # "Reuse x0 and y0" (line 74 of MATLAB) -> Refers to ORIGINAL x (passed as var, but x was modified? No, x0 variable was modified).
            # Line 74: `x0 = x;` (Original data with duplicates removed).
            # "x0(2:end) = flipud(x0(2:end))".
            # So it reorders the ORIGINAL sequence (unshifted), just reversing the path after the first point.
            
            x_out_rev = np.concatenate(([x_out[0]], x_out[1:][::-1]))
            y_out_rev = np.concatenate(([y_out[0]], y_out[1:][::-1]))
            x_out = x_out_rev
            y_out = y_out_rev
            
        if restore_end:
            x_out = np.concatenate((x_out, [x_out[0]]))
            y_out = np.concatenate((y_out, [y_out[0]]))
            
        return direction, x_out, y_out
