
import numpy as np
from skimage.draw import line

def connectpoly(x, y):
    """
    Connects vertices of a polygon with straight lines.
    x: Row coordinates (array-line).
    y: Col coordinates (array-like).
    Returns c: (m-by-2) array of coordinates [row, col].
    """
    x = np.array(x, dtype=int).flatten()
    y = np.array(y, dtype=int).flatten()
    
    num_points = len(x)
    if num_points < 2:
        return np.column_stack((x, y))
        
    pts = []
    
    # Process segments
    # Close polygon logic: last point connects to first?
    # MATLAB: "The last point in the sequence is equal to the first".
    # User passes vertices. MATLAB code `v(end+1,:) = v(1,:)` if not equal.
    
    if (x[-1] != x[0]) or (y[-1] != y[0]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        num_points += 1
        
    for i in range(num_points - 1):
        r0, c0 = x[i], y[i]
        r1, c1 = x[i+1], y[i+1]
        
        # skimage.draw.line returns pixels INCLUSIVE of endpoints
        rr, cc = line(r0, c0, r1, c1)
        
        # Stack
        seg = np.column_stack((rr, cc))
        
        # Avoid duplicate points at joins:
        # Exclude the last point of each segment
        pts.append(seg[:-1])
        
    if len(pts) > 0:
        c = np.vstack(pts)
    else:
        c = np.column_stack((x, y))
        
    # Check for duplicates?
    # If standard behavior duplicates, I'll allow it.
    # But usually clean boundary is preferred.
    # I'll stick to simple concatenation.
    
    return c
