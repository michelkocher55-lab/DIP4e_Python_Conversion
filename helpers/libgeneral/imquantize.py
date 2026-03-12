from typing import Any
import numpy as np


def imquantize(A: Any, levels: Any, values: Any = None):
    """
    Quantize image using specified quantization levels (thresholds).

    Parameters:
    A: Input image.
    levels: List/Array of thresholds (must be sorted).
    values: Optional. List/Array of N+1 values to verify/map to.
            If None, returns indices 1..N+1 (matching MATLAB behavior).
            numpy.digitize returns 0..N.

    Returns:
    Q: Quantized image.
    """
    levels = np.sort(levels)

    # np.digitize:
    # bins[i-1] <= x < bins[i]
    # If x < bins[0] -> 0
    # If x >= bins[-1] -> len(bins)

    # MATLAB imquantize(A, levels)
    # levels = [T1, T2]
    # x <= T1 -> 1
    # T1 < x <= T2 -> 2
    # x > T2 -> 3

    # np.digitize behavior strictly less than or greater equal?
    # default right=False: bins[i-1] <= x < bins[i].
    # So if x < T1 -> 0.
    # T1 <= x < T2 -> 1.
    # x >= T2 -> 2.

    # To match MATLAB:
    # x <= T1 (Val 1)
    # x > T1 (Val 2)

    # If we use right=True:
    # bins[i-1] < x <= bins[i].
    # x <= T1 -> 1 (index 1 in 1-based logic? digitize returns 1 if x <= bins[0]?)
    # CHECK DOCS:
    # right=False (default): bins[i-1] <= x < bins[i]
    #   x < bins[0] --> 0
    #   bins[0] <= x < bins[1] --> 1

    # right=True: bins[i-1] < x <= bins[i]
    #   x <= bins[0] --> 0
    #   bins[0] < x <= bins[1] --> 1
    #   x > bins[-1] --> len(bins)

    # So with right=True:
    # x <= T1 -> 0.
    # T1 < x <= T2 -> 1.
    # x > Tn -> N.

    # If we add 1 to the result:
    # 0 -> 1. (x <= T1)
    # 1 -> 2. (T1 < x <= T2)
    # N -> N+1. (x > Tn)

    # This matches MATLAB (1, 2, ... N+1).

    indices = np.digitize(A, levels, right=True)

    # Indices are 0..N.
    # Add 1 to match MATLAB 1..N+1 default output
    Q = indices + 1

    if values is not None:
        # Map indices (1-based) to values.
        # values should have N+1 elements.
        values = np.array(values)
        if len(values) != len(levels) + 1:
            raise ValueError("Number of values must be len(levels) + 1")

        # Use simple indexing
        # indices 0..N maps to values[0]..values[N]
        # Q indices were indices+1.
        # So we can just use indices to index into values.
        Q = values[indices]

    return Q
