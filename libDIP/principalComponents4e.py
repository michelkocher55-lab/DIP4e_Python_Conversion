from typing import Any
import numpy as np


def covmatrix4e(X: Any):
    """
    Computes the covariance matrix C and mean vector M of a vector population.
    X: K-by-N matrix (K samples, N dimensionality).
    Returns C (N-by-N), m (N-by-1).
    Normalization by K-1 (unbiased).
    """
    X = np.array(X, dtype=float)
    K, N = X.shape

    # Mean
    m = np.mean(X, axis=0)  # (N,)

    if K == 1:
        C = np.full((N, N), np.nan)
        return C, m.reshape(-1, 1)

    # Covariance
    # np.cov expects rows as variables (dimensions), columns as observations.
    # X has rows as observations. So X.T.
    C = np.cov(X, rowvar=False, ddof=1)

    return C, m.reshape(-1, 1)


class PrincipalComponentsResult:
    def __init__(self):
        """__init__."""
        self.Y = None
        self.A = None
        self.X = None
        self.ems = 0
        self.Cx = None
        self.mx = None
        self.Cy = None


def principalComponents4e(X: Any, q: Any):
    """
    Computes Principal-component vectors.
    X: K-by-N matrix. K vectors of dimension N.
    q: Number of eigenvectors used.
    Returns PrincipalComponentsResult object.
    """
    X = np.array(X, dtype=float)
    K, N = X.shape

    P = PrincipalComponentsResult()

    # 1. Covariance values
    P.Cx, P.mx = covmatrix4e(X)
    # P.mx is N-by-1

    # 2. Eigenvectors of Cx
    # eig in numpy: w, v = np.linalg.eig(Cx)
    vals, vecs = np.linalg.eig(P.Cx)

    # 3. Sort eigenvalues decreasing
    idx = np.argsort(vals)[::-1]
    d = vals[idx]
    V = vecs[:, idx]

    # 4. Form A from first q columns of V
    # A should be q-by-n.
    # V is n-by-n.
    # A rows are eigenvectors.
    P.A = V[:, :q].T  # (q, N)

    # 5. Compute principal component vectors Y
    # Broadcast P.mx.T (1, N)
    X_centered = X - P.mx.T
    P.Y = P.A @ X_centered.T  # (q, K)

    # 6. Reconstruct X
    # P.X = (P.A' * P.Y)' + Mx
    P.X = (P.A.T @ P.Y).T + P.mx.T

    # 7. Convert P.Y to K-by-q and P.mx to N-by-1
    P.Y = P.Y.T
    # P.mx is already N-by-1

    # 8. EMS
    # Sum of all eigenvalues minus sum of q largest.
    P.ems = np.sum(d[q:])

    # 9. Covariance matrix of Y
    # P.Cy = P.A * P.Cx * P.A'
    P.Cy = P.A @ P.Cx @ P.A.T

    return P
