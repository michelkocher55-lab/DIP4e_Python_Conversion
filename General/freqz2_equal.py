from typing import Any
import numpy as np


def freqz2_equal(h: Any, ny: Any, nx: Any, dx: Any = 0.5, dy: Any = 0.5):
    """freqz2_equal."""
    a = np.array(h, dtype=float)
    # Unrotate filter since FIR filters are rotated
    a = np.rot90(a, -2)

    m, n = a.shape
    center_a_row = int(np.ceil((m + 1) / 2.0)) - 1
    center_a_col = int(np.ceil((n + 1) / 2.0)) - 1

    # Pad if needed
    if m < ny or n < nx:
        a_pad = np.zeros((ny, nx), dtype=float)
        a_pad[:m, :n] = a
        a = a_pad
        m, n = a.shape

    # Circular shift to put center at (0,0)
    row_idx = list(range(center_a_row, ny)) + list(range(0, center_a_row))
    col_idx = list(range(center_a_col, nx)) + list(range(0, center_a_col))
    a = a[np.ix_(row_idx, col_idx)]

    H = np.fft.fftshift(np.fft.fft2(a))
    return H
