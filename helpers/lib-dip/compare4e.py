import numpy as np

def compare4e(f1, f2):
    """
    Computes the root-mean-square error (RMSE) between two matrices.
    
    Parameters:
    -----------
    f1 : numpy.ndarray
        First input matrix/image.
    f2 : numpy.ndarray
        Second input matrix/image.
        
    Returns:
    --------
    rmse : float
        Root-mean-square error.
    """
    f1 = np.array(f1, dtype=float)
    f2 = np.array(f2, dtype=float)
    
    if f1.shape != f2.shape:
        raise ValueError("Input matrices must have the same dimensions.")
        
    # Difference
    e = f1 - f2
    
    # RMSE
    # sqrt(mean(e^2))
    mse = np.mean(e**2)
    rmse = np.sqrt(mse)
    
    return rmse
