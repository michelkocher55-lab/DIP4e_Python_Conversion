from typing import Any
import numpy as np
from libDIP.mahalanobisDistance4e import mahalanobisDistance4e


def bayesClassifier4e(X: Any, CA: Any, MA: Any, P: Any = None):
    """
    Bayes classifier for Gaussian patterns.

    Computes the Bayes decision functions of the n-dimensional patterns in the
    rows of X.

    Parameters:
    -----------
    X : numpy.ndarray
        (np, n) array where each row is a sample pattern to classify.
    CA : list of numpy.ndarray or numpy.ndarray
        Covariance matrices for each class.
        If list: len(nc), each element is (n, n).
        If array: (nc, n, n) or (n, n, nc) - to match MATLAB's (n, n, nc),
                  we will check dimensions but prefer (nc, n, n) in Python.
    MA : list of numpy.ndarray or numpy.ndarray
        Mean vectors for each class.
        If list: len(nc), each element is (n,).
        If array: (nc, n).
    P : list or numpy.ndarray, optional
        (nc,) array of prior probabilities for each class.
        If None, classes are assumed to be equally likely.

    Returns:
    --------
    d : numpy.ndarray
        (np,) array containing the class index (0 to nc-1) assigned to each
        vector in X.
    """
    X = np.atleast_2d(X)
    np_samples, n = X.shape

    # Handle covariance matrices input
    if isinstance(CA, list):
        CA = np.array(CA)

    # Check CA shape. MATLAB uses (n, n, nc). Python usually (nc, n, n).
    # We prioritize (nc, n, n).
    if CA.ndim != 3:
        raise ValueError("CA must be a 3D array or list of matrices.")

    if CA.shape[1] == n and CA.shape[2] == n:
        # Standard Python shape (nc, n, n)
        # Even if n == nc, we assume this format if dimensions match
        pass
    elif CA.shape[0] == n and CA.shape[1] == n:
        # MATLAB shape (n, n, nc)
        CA = np.transpose(CA, (2, 0, 1))
    else:
        raise ValueError(
            f"CA shape {CA.shape} does not match expected dimensions for n={n}."
        )

    nc = CA.shape[0]  # number of classes

    # Handle Mean Vectors
    if isinstance(MA, list):
        MA = np.array(MA)

    # Expected MA shape: (nc, n). If (n, nc), transpose.
    if MA.shape != (nc, n):
        if MA.shape == (n, nc):
            MA = MA.T
        elif MA.shape == (n,):  # Single class? or just weird
            MA = MA.reshape(1, n)
        else:
            pass  # hope for the best or raise error

    if MA.shape[0] != nc:
        raise ValueError(
            f"Number of mean vectors {MA.shape[0]} does not match number of covariance matrices {nc}."
        )

    # Handle Priors P
    if P is None:
        P = np.ones(nc) / nc
    else:
        P = np.array(P).flatten()
        if len(P) != nc:
            raise ValueError(
                f"Number of priors {len(P)} does not match number of classes {nc}."
            )
        if not np.isclose(np.sum(P), 1.0):
            raise ValueError("Elements of P must sum to 1.")

    # Compute determinants and log-determinants
    # DM[j] = det(CA[j])
    # Use slogdet for stability? MATLAB uses det.
    # log(det) is needed.
    log_dets = np.zeros(nc)
    for j in range(nc):
        # sign, logdet = np.linalg.slogdet(CA[j])
        # if sign <= 0: raise ValueError("Covariance matrix must be positive definite.")
        # log_dets[j] = logdet
        # Simpler approach matching MATLAB directly:
        det_val = np.linalg.det(CA[j])
        if det_val <= 0:
            # Handle singular or negative det if possible, or produce -inf
            log_dets[j] = -np.inf
        else:
            log_dets[j] = np.log(det_val)

    # Decision functions
    # d_j(x) = ln(P(w_j)) - 0.5*ln(|C_j|) - 0.5 * (x-m_j)^T * inv(C_j) * (x-m_j)

    D_vals = np.zeros((np_samples, nc))

    for j in range(nc):
        if P[j] == 0:
            D_vals[:, j] = -np.inf
            continue

        L = np.log(P[j])
        DET_term = 0.5 * log_dets[j]

        # Mahalanobis distance squared
        # use our function: (np, )
        dist_sq = mahalanobisDistance4e(X, cov=CA[j], mean=MA[j])

        # Calculate discriminant function value for all patterns for class j
        # MATLAB: D(:,J) = L - DET - 0.5*mahalanobisDistance4e(X,C,M);
        D_vals[:, j] = L - DET_term - 0.5 * dist_sq

    # Assign class with maximum discriminant function value
    # Python argmax returns 0-based index.
    d = np.argmax(D_vals, axis=1)

    return d
