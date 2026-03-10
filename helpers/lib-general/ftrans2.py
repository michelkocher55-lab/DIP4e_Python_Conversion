from typing import Any
import numpy as np
from scipy.signal import convolve2d


def ftrans2(b: Any, t: Any = None):
    """
    FTRANS2 2-D FIR filter using frequency transformation.

    h = ftrans2(b, t=None)

    Parameters
    ----------
    b : array-like
        1-D type I FIR filter (odd length, symmetric).
    t : array-like, optional
        2-D transformation matrix. Default is McClellan transform.

    Returns
    -------
    h : ndarray
        2-D FIR computational molecule (rotated for filter2 usage).
    """
    b = np.asarray(b, dtype=float)
    if b.size == 0 or np.all(b == 0):
        raise ValueError("B must contain at least one nonzero element.")

    if t is None:
        t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]], dtype=float) / 8.0
    else:
        t = np.asarray(t, dtype=float)
        if t.size == 0 or np.all(t == 0):
            raise ValueError("T must contain at least one nonzero element.")

    if b.ndim != 1 and not (b.ndim == 2 and (1 in b.shape)):
        raise ValueError("B must be a 1-D vector.")
    b = b.ravel()

    n = (len(b) - 1) / 2.0
    if int(np.floor(n)) != n:
        raise ValueError("b must be odd length.")
    n = int(n)

    if not np.allclose(b, b[::-1], atol=np.sqrt(np.finfo(float).eps)):
        raise ValueError("b must be symmetric.")

    # MATLAB: b = rot90(fftshift(rot90(b,2)),2); % inverse fftshift in 1-D
    b = np.fft.ifftshift(b)

    # Convert 1-D filter to sum_n a(n) cos(w n) form
    a = np.concatenate(([b[0]], 2.0 * b[1 : n + 1]))

    inset = ((np.array(t.shape) - 1) // 2).astype(int)

    # Chebyshev recursion
    P0 = np.array([[1.0]])
    P1 = t.copy()

    h = a[1] * P1

    # MATLAB rows, cols start as scalar center location
    rows_idx = np.array([inset[0]], dtype=int)
    cols_idx = np.array([inset[1]], dtype=int)
    h[np.ix_(rows_idx, cols_idx)] += a[0] * P0

    for i in range(2, n + 1):
        P2 = 2.0 * convolve2d(t, P1, mode="full")

        # rows = rows + inset; cols = cols + inset;
        rows_idx = rows_idx + inset[0]
        cols_idx = cols_idx + inset[1]

        # P2(rows, cols) = P2(rows, cols) - P0;
        P2[np.ix_(rows_idx, cols_idx)] -= P0

        # rows = inset + (1:size(P1,1)); cols = inset + (1:size(P1,2));
        rows_idx = inset[0] + np.arange(P1.shape[0])
        cols_idx = inset[1] + np.arange(P1.shape[1])

        hh = h
        h = a[i] * P2
        h[np.ix_(rows_idx, cols_idx)] += hh

        P0 = P1
        P1 = P2

    # Rotate for use with filter2
    h = np.rot90(h, 2)
    return h
