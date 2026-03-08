
import numpy as np
from libDIPUM.interparc import interparc

def snakeReparam4e(x, y):
    """
    SNAKEREPARAM4E Reparameterizes a snake.
    Literal transcoding of snakeReparam4e.m.
    """
    
    # -Number of input points.
    # np = numel(x);
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    np_pts = len(x)
    
    # -The following function performs interpolation...
    # pt = interparc(np,y,x,'linear');
    # Note: MATLAB interparc(t, px, py). 
    # Passes y(Col) as px, x(Row) as py.
    # Python interparc wrapper should handle this.
    # returns pt array where col 1 is px(y), col 2 is py(x).
    
    # Python interparc returns tuple (qx, qy) corresponding to inputs.
    # If we pass (y, x), we get (qy, qx).
    # qy matches input y (Col). qx matches input x (Row).
    
    qy, qx = interparc(np_pts, y, x)
    
    # -Extract the coordinates from the output, pt.
    # --Row coordinates.
    # xp = pt(:,2);
    # corresponds to qx above.
    xp = qx
    
    # --Column coordinates
    # yp = pt(:,1);
    # corresponds to qy above.
    yp = qy
    
    # -Make sure the curve is closed.
    # xp(end) = xp(1);
    # yp(end) = yp(1);
    xp[-1] = xp[0]
    yp[-1] = yp[0]
    
    return xp, yp
