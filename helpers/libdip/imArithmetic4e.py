import numpy as np

def imArithmetic4e(f1, f2, op):
    """
    Arithmetic operations between two grayscale images.
    
    Parameters:
    -----------
    f1 : numpy.ndarray
        First input image.
    f2 : numpy.ndarray
        Second input image.
    op : str
        Operation: 'add', 'subtract', 'multiply', 'divide'.
        
    Returns:
    --------
    g : numpy.ndarray
        Result image (floating point).
    """
    # Convert inputs to floating point
    f1 = np.array(f1, dtype=float)
    f2 = np.array(f2, dtype=float)
    
    op = op.lower()
    
    if op == 'add':
        g = f1 + f2
    elif op == 'subtract':
        g = f1 - f2
    elif op == 'multiply':
        g = f1 * f2
    elif op == 'divide':
        # Add eps to prevent division by 0
        eps = np.finfo(float).eps
        g = f1 / (f2 + eps)
    else:
        raise ValueError(f"Unknown operation: {op}")
        
    return g
