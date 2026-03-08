import numpy as np

def div2d(gx, gy):
    """
    Computes the divergence of a 2D vector field.
    
    Parameters:
    -----------
    gx : numpy.ndarray
        x-component (Vertical/Row) of vector field.
    gy : numpy.ndarray
        y-component (Horizontal/Col) of vector field.
        
    Returns:
    --------
    D : numpy.ndarray
        Divergence field. D = d(gx)/dx + d(gy)/dy.
        (Note: following MATLAB notation where gx is row-derivative)
    """
    gx = gx.astype(float)
    gy = gy.astype(float)
    
    # Pad inputs
    # MATLAB: gxp = padarray(gx,[1,0],'symmetric','both'); (Pad Rows)
    gxp = np.pad(gx, ((1, 1), (0, 0)), mode='symmetric')
    
    # MATLAB: gyp = padarray(gy,[0,1],'symmetric','both'); (Pad Cols)
    gyp = np.pad(gy, ((0, 0), (1, 1)), mode='symmetric')
    
    # Central Differences
    # MATLAB: dx(1:M,:) = (gxp(3:M+2,:) - gxp(1:M,:))/2;
    # Slice [2:] - Slice [:-2]
    dx = (gxp[2:, :] - gxp[:-2, :]) / 2.0
    
    # MATLAB: dy(:,1:N) = (gyp(:,3:N+2) - gyp(:,1:N))/2;
    dy = (gyp[:, 2:] - gyp[:, :-2]) / 2.0
    
    D = dx + dy
    
    return D
