from typing import Any
import numpy as np


def covmatrix4e(X: Any):
    """
    Computes the covariance matrix and mean vector.

    Parameters:
    -----------
    X : numpy.ndarray
        A K-by-N matrix where K is the number of samples and N is their
        dimensionality.

    Returns:
    --------
    C : numpy.ndarray
        N-by-N covariance matrix.
    m : numpy.ndarray
        N-by-1 mean vector.
    """
    X = np.array(X, dtype=float)

    K, N = X.shape

    # Compute unbiased estimate of m
    # MATLAB: m = sum(X, 1)/K; -> Row vector 1xN
    # Python: mean over axis 0 -> vector of size N
    m = np.mean(X, axis=0)  # shape (N,)

    # Subtract the mean from each row of X
    # MATLAB: X = X - m(ones(K, 1), :);
    # Python: Broadcasting handles X - m (N,) correctly for (K, N) array
    X_centered = X - m

    # Compute unbiased estimate of C.
    if K > 1:
        # MATLAB: C = (X'*X)/(K - 1);
        # Python: X.T @ X / (K - 1)
        C = (X_centered.T @ X_centered) / (K - 1)
    else:
        # If population contains a single sample, return C as NaNs
        C = np.full((N, N), np.nan)

    # Convert m to column vector
    m = m.reshape(N, 1)

    return C, m
