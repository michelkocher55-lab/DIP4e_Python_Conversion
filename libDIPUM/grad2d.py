import numpy as np

def grad2d(f):
    """
    Computes the gradient of a 2-D array.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input 2-D array.
        
    Returns:
    --------
    gx : numpy.ndarray
        Gradient in x-direction (Rows/Vertical).
        Corresponds to d/dr.
    gy : numpy.ndarray
        Gradient in y-direction (Cols/Horizontal).
        Corresponds to d/dc.
    """
    f = f.astype(float)
    M, N = f.shape
    
    # Pad borders
    # MATLAB: padarray(f, [1, 0], 'symmetric', 'both'); -> Pads rows (Top/Bottom)
    fx = np.pad(f, ((1, 1), (0, 0)), mode='symmetric')
    
    # MATLAB: padarray(f, [0, 1], 'symmetric', 'both'); -> Pads cols (Left/Right)
    fy = np.pad(f, ((0, 0), (1, 1)), mode='symmetric')
    
    # Central Differences
    # MATLAB: gx(1:M,:) = (fx(3:M+2,:) - fx(1:M,:))/2;
    # fx Indices: 0, 1, ..., M+1 (Length M+2)
    # 3:M+2 (1-based) -> 2:M+1 (0-based) -> Slice [2:]
    # 1:M (1-based) -> 0:M-1 (0-based) -> Slice [:-2]
    gx = (fx[2:, :] - fx[:-2, :]) / 2.0
    
    # MATLAB: gy(:,1:N) = (fy(:,3:N+2) - fy(:,1:N))/2;
    # fy Indices: 0, 1, ..., N+1 (Length N+2)
    gy = (fy[:, 2:] - fy[:, :-2]) / 2.0
    
    return gx, gy
