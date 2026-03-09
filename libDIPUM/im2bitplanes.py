from typing import Any
import numpy as np


def im2bitplanes(f: Any, n: Any = None):
    """
    Extracts all the bit planes of an image.

    Parameters:
    f (ndarray): Input image (integer type expected).
    n (int, optional): Number of bits. If None, inferred from dtype (8 for uint8, 16 for uint16).

    Returns:
    B (ndarray): Bit planes array of shape (M, N, n).
                 B[:,:,0] is the Least Significant Bit (LSB).
                 B[:,:,n-1] is the Most Significant Bit (MSB).
                 Data type is bool (logical).
    """
    f = np.asarray(f)

    if n is None:
        if f.dtype == np.uint8:
            n = 8
        elif f.dtype == np.uint16:
            n = 16
        else:
            # Default to 8 or raise error? MATLAB documentation says input should be uint8 or uint16.
            # If double, values must be in range.
            # We'll assume integer values.
            n = 8

    # Convert to integer type if float, to support bitwise operations
    # MATLAB code does double arithmetic. We can cast to int for bitwise ops.
    if np.issubdtype(f.dtype, np.floating):
        f_int = f.astype(np.uint64)  # Use 64 to be safe
    else:
        f_int = f

    M, N = f.shape[:2]
    B = np.zeros((M, N, n), dtype=bool)

    for k in range(n):
        # Check k-th bit (0-based)
        # B[:,:,k] = (f >> k) & 1
        B[:, :, k] = (f_int >> k) & 1

    return B
