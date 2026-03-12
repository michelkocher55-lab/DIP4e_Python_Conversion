from typing import Any
import numpy as np
from scipy import signal


def movingthresh(f: Any, n: Any, K: Any):
    """
    Image segmentation using a moving average threshold.

    Parameters:
        f: Input image (2D array).
        n: Number of points in moving average (integer >= 1).
        K: Fraction for thresholding [0, 1].

    Returns:
        g: Segmented image (boolean).
    """
    # Validate inputs
    if n < 1 or int(n) != n:
        raise ValueError("n must be an integer >= 1.")
    if K < 0 or K > 1:
        raise ValueError("K must be a fraction in the range [0, 1].")

    f = np.array(f, dtype=np.float64)
    M, N = f.shape

    # Flip every other row to produce zig-zag pattern equivalent
    # Python 0-based indexing: Odd rows (indices 1, 3, 5...) correspond to MATLAB even rows (2, 4, 6...)
    # MATLAB: f(2:2:end, :) = fliplr(f(2:2:end, :))
    # Python: f[1::2, :] = np.fliplr(f[1::2, :])

    # We work on a copy to avoid modifying input in-place if passed by reference
    f_processed = f.copy()
    f_processed[1::2, :] = np.fliplr(f_processed[1::2, :])

    # Convert to 1D array
    # MATLAB: f = f' -> f = f(:)' (column-wise flatten then transpose?)
    # Wait, MATLAB ' operator on matrix is conjugate transpose.
    # MATLAB: f = f'. Transposes M x N to N x M.
    # Then f = f(:)'. Flattens column-wise.
    # Effectively, it reads row 1 then row 2... ?
    # Let's trace MATLAB:
    # A = [1 2; 3 4]
    # A(2:2:end,:) = fliplr... -> [1 2; 4 3]
    # A = A' -> [1 4; 2 3]
    # A(:)' -> [1 2 4 3]
    # So it just reads row by row (1,2) then (4,3).
    # Python flatten() is row-major (C-style) by default.
    # So A.flatten() on [1 2; 4 3] is [1 2 4 3].
    # So matches perfectly.

    f_flat = f_processed.flatten()

    # Compute moving average
    # maf = ones(1, n)/n
    # ma = filter(maf, 1, f)
    # MATLAB filter(b, a, x) implements difference equation.
    # y[k] = b[0]*x[k] + ... + b[nb]*x[k-nb] - ...
    # Here b = ones(n)/n. a=1.
    # y[k] = (1/n) * sum(x[k-i]) for i=0..n-1.
    # Causal moving average.

    b = np.ones(int(n)) / n
    a = 1

    # scipy.signal.lfilter matches MATLAB filter
    ma = signal.lfilter(b, a, f_flat)

    # Thresholding
    # g = f > K * ma
    g_flat = f_flat > (K * ma)

    # Go back to image format
    # MATLAB: g = reshape(g, N, M)'
    # MATLAB reshape is column-major.
    # g (1D) length M*N.
    # reshape(g, N, M) fills columns first.
    # Then transpose.
    # Effectively fills row by row if we just reshape(M, N) in row-major?
    # Let's check:
    # 1D: [1 2 4 3]
    # MATLAB reshape([1 2 4 3], 2, 2) ->
    # Col 1: 1, 2. Col 2: 4, 3.
    # [1 4; 2 3]
    # Transpose -> [1 2; 4 3].
    # Original processed matrix was [1 2; 4 3].
    # So yes, MATLAB sequence: flatten row-wise -> process -> reshape row-wise back.
    # Python reshape is row-major by default.
    # g_flat.reshape(M, N) -> [[1, 2], [4, 3]].
    # Matches.

    g = g_flat.reshape(M, N)

    # Flip alternate rows back
    g[1::2, :] = np.fliplr(g[1::2, :])

    return g
