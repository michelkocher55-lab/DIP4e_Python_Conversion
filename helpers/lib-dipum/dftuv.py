from typing import Any
import numpy as np


def dftuv(M: Any, N: Any):
    """
    Computes meshgrid frequency matrices.

    Parameters:
    M, N (int): Size of the frequency rectangle.

    Returns:
    U, V (ndarray): Frequency coordinates of size MxN.
    """
    # Set up range of variables.
    u = np.arange(M)
    v = np.arange(N)

    # Compute the indices for use in meshgrid.
    # idx = find(u > M/2);
    # u(idx) = u(idx) - M;
    # In Python: u[u > M/2] -= M
    # Use numpy boolean indexing

    # MATLAB: idx = find(u > M/2). If M=4, u=[0,1,2,3]. M/2=2. u>2 -> 3. u[3]=3-4=-1. Result: 0, 1, 2, -1.
    # MATLAB: idx = find(u > M/2). If M=5, u=[0,1,2,3,4]. M/2=2.5. u>2.5 -> 3,4. u[3]=-2, u[4]=-1. Result: 0, 1, 2, -2, -1.

    # Wait, simple frequency wraparound logic is usually:
    # if k < N/2: k
    # else: k - N
    # This matches exactly.

    idx = u > M / 2
    u[idx] = u[idx] - M

    idy = v > N / 2
    v[idy] = v[idy] - N

    # Compute the meshgrid arrays.
    # [V, U] = meshgrid(v, u);
    # In Python np.meshgrid(v, u) returns V, U (where V has shape (N,M)? No, numpy meshgrid defaults to 'xy' indexing).
    # 'xy' means: x (col), y (row).
    # np.meshgrid(v, u) -> X coords from v (cols change), Y coords from u (rows change).
    # So output matrices are (len(u), len(v)) = (M, N).
    # First output is X (from v, changes along cols). Second output is Y (from u, changes along rows).
    # So `V, U = np.meshgrid(v, u)` is correct naming relative to axis.

    V, U = np.meshgrid(v, u)

    return U.astype(float), V.astype(float)
