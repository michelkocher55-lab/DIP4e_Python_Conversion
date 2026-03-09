from typing import Any
import numpy as np
from skimage.filters import threshold_multiotsu


def multithresh(A: Any, N: Any):
    """
    Returns N thresholds for image A using Otsu's method.

    Parameters:
    A: Input image.
    N: Number of thresholds.

    Returns:
    T: Array of N thresholds.
    """
    # skimage threshold_multiotsu returns N thresholds for classes=N+1
    try:
        T = threshold_multiotsu(A, classes=N + 1)
    except ValueError:
        # Fallback if image has fewer unique values than classes
        # or other issues.
        # Just return linspace?
        mi, ma = A.min(), A.max()
        T = np.linspace(mi, ma, N + 2)[1:-1]

    return T
