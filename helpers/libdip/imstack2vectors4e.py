import numpy as np

def imstack2vectors4e(S, mask=None):
    """
    Extracts vectors from an image stack.

    Parameters:
    -----------
    S : numpy.ndarray
        An (M, N, n) stack of n registered images of size M-by-N each.
    mask : numpy.ndarray, optional
        An (M, N) logical or numeric image with nonzero values (True if boolean)
        in locations where elements of S are to be used.
        If omitted, all M*N locations are used.

    Returns:
    --------
    X : numpy.ndarray
        Array of shape (K, n), where K is the number of nonzero elements in mask.
        Each row is a vector extracted from S.
    R : numpy.ndarray
        A 1D array containing the linear indices (flat indices) of the locations
        of the vectors extracted from S.
    """
    # Check input dimensions
    if S.ndim != 3:
        raise ValueError("Input S must be a 3D array (M, N, n).")
    
    M, N, n = S.shape

    # Handle mask
    if mask is None:
        mask = np.ones((M, N), dtype=bool)
    else:
        # Ensure mask is 2D and matches S dimensions
        if mask.shape != (M, N):
             raise ValueError(f"Mask shape {mask.shape} does not match image dimensions {(M, N)}.")
        mask = (mask != 0)

    # Flatten S to (M*N, n) using row-major order (standard Python/NumPy order)
    # Note: MATLAB uses column-major order. If strict MATLAB compatibility for
    # index order 'R' is required, we would need to flatten with order='F'.
    # However, for Python usage, the standard is C-style (row-major).
    # The user was notified about this difference in the plan.
    
    # Reshape S to (M*N, n). 
    # Logic: We want each pixel location (i, j) to be a row in X.
    # S[i, j, :] gives the vector for pixel (i, j).
    # reshaping with default order='C' (row-major) iterates over last axis fastest?
    # No, reshape((M*N, n)) on (M, N, n) simply flattens the first two dims if we are careful.
    
    # Actually, a safer way to be explicit regardless of memory layout:
    # reshape(-1, n) works if the data is stored (M, N, n).
    # It will walk through M, then N.
    X_full = S.reshape(-1, n) 
    
    # Flatten MASK to align with X_full
    mask_flat = mask.reshape(-1)
    
    # Find indices where mask is True
    # nonzero returns a tuple of arrays, one for each dimension. 
    # Since mask_flat is 1D, it returns (indices_array,)
    R = np.nonzero(mask_flat)[0]
    
    # Filter X
    X = X_full[mask_flat, :]
    
    return X, R
