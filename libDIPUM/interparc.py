
import numpy as np

def interparc(t, px, py):
    """
    Minimal implementation of interparc for linear interpolation.
    Interpolates independent x and y coordinates as a function of arc length.
    
    Parameters:
        t: Number of points or array of points.
        px: x coordinates (columns)
        py: y coordinates (rows)
    
    Returns:
        qx, qy: Interpolated coordinates.
    """
    # Compute cumulative linear distance
    px = np.asarray(px)
    py = np.asarray(py)
    
    dx = np.diff(px)
    dy = np.diff(py)
    
    # Chord lengths
    dist = np.sqrt(dx**2 + dy**2)
    
    # Cumulative distance
    cumdist = np.concatenate(([0], np.cumsum(dist)))
    
    # Normalize to 0-1
    total_dist = cumdist[-1]
    if total_dist == 0:
         # All points same
         return px, py
         
    t_norm = cumdist / total_dist
    
    # Desired points
    # if t is integer, generate t equally spaced points
    if isinstance(t, int):
        t_query = np.linspace(0, 1, t)
    else:
        t_query = np.asarray(t)
        
    # Interpolate x and y separately vs t_norm
    qx = np.interp(t_query, t_norm, px)
    qy = np.interp(t_query, t_norm, py)
    
    return qx, qy
