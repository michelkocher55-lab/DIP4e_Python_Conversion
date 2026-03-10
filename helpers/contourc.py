from typing import Any
import numpy as np
from skimage.measure import find_contours


def contourc(Z: Any, levels: Any):
    """
    Approximate MATLAB contourc for a single level.

    Returns a 2xN array with:
      row 0 = x (columns)
      row 1 = y (rows)

    This is sufficient for DIP4e scripts that use contourc(phi, [0 0]).
    """
    if isinstance(levels, (list, tuple, np.ndarray)):
        if len(levels) == 0:
            level = 0.0
        else:
            level = float(levels[0])
    else:
        level = float(levels)

    contours = find_contours(Z, level=level)
    if len(contours) == 0:
        return np.zeros((2, 0), dtype=float)

    # Use the longest contour for consistency
    longest = max(contours, key=lambda c: c.shape[0])
    # find_contours returns (row, col)
    y = longest[:, 0]
    x = longest[:, 1]

    return np.vstack([x, y])
