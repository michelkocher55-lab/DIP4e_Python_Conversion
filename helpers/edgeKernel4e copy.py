
import numpy as np

def edgeKernel4e(type_='sobel', dir_='v'):
    """
    Generates a 3x3 edge kernel.
    
    Parameters:
        type_: 'prewitt', 'sobel', 'kirsch'.
        dir_: Direction string.
            prewitt/sobel: 'v', 'h'.
            kirsch: 'n', 'nw', 'w', 'sw', 's', 'se', 'e', 'ne'.
            
    Returns:
        w: 3x3 kernel.
    """
    type_ = type_.lower()
    dir_ = dir_.lower()
    
    w = np.zeros((3, 3))
    
    if type_ == 'prewitt':
        if dir_ == 'v':
            w = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        elif dir_ == 'h':
            # MATLAB: [-1 -1 -1; 0 0 0; 1 1 1]
            w = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            
    elif type_ == 'sobel':
        if dir_ == 'v':
            w = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif dir_ == 'h':
            w = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
    elif type_ == 'kirsch':
        if dir_ == 'n':
            w = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
        elif dir_ == 'nw':
            w = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
        elif dir_ == 'w':
            w = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
        elif dir_ == 'sw':
             w = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
        elif dir_ == 's':
            w = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
        elif dir_ == 'se':
            w = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
        elif dir_ == 'e':
            w = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
        elif dir_ == 'ne':
             w = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    
    return w
