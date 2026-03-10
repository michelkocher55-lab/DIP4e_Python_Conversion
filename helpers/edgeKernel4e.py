import numpy as np

def edgeKernel4e(type, dir):
    """
    Generates a 3-by-3 edge kernel of specified TYPE and direction (DIR).

    Parameters:
    -----------
    type : str
        'prewitt', 'sobel', or 'kirsch'.
    dir : str
        Direction.
        For 'prewitt' and 'sobel': 'v' (vertical) or 'h' (horizontal).
        For 'kirsch': 'n', 'nw', 'w', 'sw', 's', 'se', 'e', 'ne'.

    Returns:
    --------
    w : numpy.ndarray
        3x3 edge kernel.
    """
    w = None
    
    if type == 'prewitt':
        if dir == 'v':
            w = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
        elif dir == 'h':
            w = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]])
                          
    elif type == 'sobel':
        if dir == 'v':
            w = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
        elif dir == 'h':
            w = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]])
                          
    elif type == 'kirsch':
        if dir == 'n':
            w = np.array([[-3, -3, 5],
                          [-3, 0, 5],
                          [-3, -3, 5]])
        elif dir == 'nw':
            w = np.array([[-3, 5, 5],
                          [-3, 0, 5],
                          [-3, -3, -3]])
        elif dir == 'w':
            w = np.array([[5, 5, 5],
                          [-3, 0, -3],
                          [-3, -3, -3]])
        elif dir == 'sw':
            w = np.array([[5, 5, -3],
                          [5, 0, -3],
                          [-3, -3, -3]])
        elif dir == 's':
            w = np.array([[5, -3, -3],
                          [5, 0, -3],
                          [5, -3, -3]])
        elif dir == 'se':
            w = np.array([[-3, -3, -3],
                          [5, 0, -3],
                          [5, 5, -3]])
        elif dir == 'e':
            w = np.array([[-3, -3, -3],
                          [-3, 0, -3],
                          [5, 5, 5]])
        elif dir == 'ne':
            w = np.array([[-3, -3, -3],
                          [-3, 0, 5],
                          [-3, 5, 5]])
                          
    if w is None:
        raise ValueError(f"Unknown type '{type}' or direction '{dir}'.")
        
    return w
