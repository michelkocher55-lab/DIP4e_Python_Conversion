from typing import Any
import numpy as np


def bound2im(b: Any, M: Any = None, N: Any = None):
    """
    Converts a boundary to an image.
    b: Nx2 array of coordinates [row, col].
    M, N: Image dimensions (optional).
          If omitted, the smallest image containing the boundary is returned,
          and coordinates are shifted to start at (0, 0).
    Returns: Binary image (M x N).
    """
    b = np.array(b)

    # Round to integers
    b = np.round(b).astype(int)

    # Check input
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("The boundary must be of size np-by-2")

    if M is None or N is None:
        # Shift to 0-based origin
        min_r = np.min(b[:, 0])
        min_c = np.min(b[:, 1])

        b[:, 0] -= min_r
        b[:, 1] -= min_c

        M = np.max(b[:, 0]) + 1
        N = np.max(b[:, 1]) + 1

    # Create image
    img = np.zeros((M, N), dtype=bool)

    # Filter bounds
    valid = (b[:, 0] >= 0) & (b[:, 0] < M) & (b[:, 1] >= 0) & (b[:, 1] < N)
    b_valid = b[valid]

    if len(b_valid) < len(b):
        pass  # Warning?

    img[b_valid[:, 0], b_valid[:, 1]] = True

    return img
