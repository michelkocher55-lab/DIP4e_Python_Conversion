import numpy as np

def mahalanobisDistance4e(Y, X=None, cov=None, mean=None):
    """
    Computes the squared Mahalanobis distance.

    D = mahalanobisDistance4e(Y, X)
    Computes the squared Mahalanobis distance between each vector in Y and the
    mean (centroid) of the vectors in X, using the covariance of X.

    D = mahalanobisDistance4e(Y, cov=CX, mean=MX)
    Computes the squared Mahalanobis distance between each vector in Y and the
    given mean vector MX, using the covariance matrix CX.

    Parameters:
    -----------
    Y : numpy.ndarray
        (M, K) array where each row is a vector (observation).
    X : numpy.ndarray, optional
        (N, K) array where each row is a vector from the reference population.
        Used to compute mean and covariance if cov/mean are not provided.
    cov : numpy.ndarray, optional
        (K, K) covariance matrix.
    mean : numpy.ndarray, optional
        (K,) mean vector.

    Returns:
    --------
    D : numpy.ndarray
        (M,) array containing the squared Mahalanobis distances.
    """
    Y = np.atleast_2d(Y)

    if X is not None:
        # Compute mean and covariance from X
        X = np.atleast_2d(X)
        if cov is None:
            cov = np.cov(X, rowvar=False)
        if mean is None:
            mean = np.mean(X, axis=0)
            
    if cov is None or mean is None:
         raise ValueError("Either X must be provided, or both cov and mean must be provided.")

    # Ensure mean is a 1D array or correct shape for broadcasting
    mean = np.array(mean).flatten()
    
    # Subtract mean
    Yc = Y - mean
    
    # Compute Mahalanobis distance
    # D^2 = (y - u) * Sigma^-1 * (y - u)^T
    # We want this for each row.
    # Formula equivalent to: sum( (Yc @ inv(Cov)) * Yc, axis=1 )
    
    # Handle the scalar covariance case (1 feature)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    if cov.shape == (1,) and Y.shape[1] == 1: # weird scalar edge case in numpy
        cov = np.array([[cov[0]]])

    try:
        # Using linalg.solve is numerically more stable than inv(cov)
        # We want X * inv(cov). 
        # let A = Yc.T (K, M)
        # Solve Cov * Z = A -> Z = inv(Cov) * A
        # Then Z.T is (M, K) -> corresponding to Yc * inv(Cov)
        
        # However, numpy.linalg.solve expects (N, N) matrix and (N, M) RHS.
        # So we solve cov * Z = Yc.T
        # Z will be (K, M).
        # The term we want is Yc * inv(cov).
        # Notice that inv(cov) is symmetric.
        # So (Yc * inv(cov))' = inv(cov) * Yc'.
        # Let Z = inv(cov) * Yc'.
        # Then we want dot(Yc, Z.T) elementwise sum.
        
        # Alternatively using solve:
        # right = (Yc / Cx) in MatLab is Yc * inv(Cx)
        # Transpose: inv(Cx)' * Yc' = inv(Cx) * Yc'
        
        Z = np.linalg.solve(cov, Yc.T) # (K, M)
        
        # Now we want sum(Yc * Z.T, axis=1) -> elementwise multiplication and sum
        # Z.T is (M, K). Yc is (M, K).
        D = np.sum(Yc * Z.T, axis=1)
        
    except np.linalg.LinAlgError:
        # Fallback for singular matrix if strictly needed, or just let it raise
        # For this conversion, mirroring standard behavior (error) is appropriate.
        raise ValueError("Covariance matrix is singular.")

    return D
