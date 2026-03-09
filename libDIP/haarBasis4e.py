from typing import Any
import numpy as np
from lib.invHaar4e import invHaar4e


def haarBasis4e(N: Any = 8):
    """
    Generate the Haar basis functions of size N.

    Parameters:
    -----------
    N : int, optional
        Size of the basis (must be power of 2). Default is 8.

    Returns:
    --------
    basis : numpy.ndarray
        NxN matrix where each row i is the reconstruction of the i-th identity vector.
    """
    I = np.eye(N)
    basis = np.zeros((N, N))

    for i in range(N):
        # Reconstruct impulse i
        basis[i, :] = invHaar4e(I[i, :])

    return basis
