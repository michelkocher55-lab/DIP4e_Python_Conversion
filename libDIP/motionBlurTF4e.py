from typing import Any
import numpy as np


def motionBlurTF4e(M: Any, N: Any, a: Any, b: Any, T: Any):
    """
    Computes a motion blur transfer function.

    H = motionBlurTF4e(M, N, a, b, T)

    Implements Eq. (5-77) in DIP4E.
    The filter transfer function is centered on the frequency rectangle.
    """

    # Setup frequency range
    u_range = np.arange(M)
    v_range = np.arange(N)

    # Meshgrid (col, row convention handled by numpy meshgrid 'xy' vs 'ij')
    # MATLAB: [v, u] = meshgrid(V, U) -> V is columns (x), U is rows (y).
    # Numpy: meshgrid(x, y) returns X (rows same as x, cols vary), Y (cols same as y, rows vary).
    # X, Y = np.meshgrid(v_range, u_range)
    # X is (M, N), varies along columns (0..N-1)
    # Y is (M, N), varies along rows (0..M-1)

    # To match MATLAB `[v,u] = meshgrid(V,U)` where U is 0:M-1 (rows) and V is 0:N-1 (cols):
    v, u = np.meshgrid(v_range, u_range)

    # Center coordinates
    # MATLAB: u = u - floor((M/2) + 1);
    # Python indices are 0-based.
    # Center logic for length M: M//2
    # MATLAB (1-based): ceil((M+1)/2) or floor(M/2)+1 is center index.
    # Standard centering: M/2 (approx).
    # Let's stick to the math: DC component should be at center.
    # MATLAB `u` becomes -M/2 ... M/2 - 1 (or similar).

    u = u - M // 2
    v = v - N // 2
    # Note: Standard centering for fftshift (DC at M//2)
    # Checks:
    # M=4: 0,1,2,3 -> -2,-1,0,1. fftshift -> 0,1,-2,-1 (DC at 0). Correct.
    # M=5: 0,1,2,3,4 -> -2,-1,0,1,2. fftshift -> 0,1,2,-2,-1 (DC at 0). Correct.

    # Eq (5-77)
    den = np.pi * (a * u + b * v) + np.finfo(float).eps
    first_term = T / den
    exp_term = -1j * den

    H = first_term * np.sin(den) * np.exp(exp_term)

    return H
