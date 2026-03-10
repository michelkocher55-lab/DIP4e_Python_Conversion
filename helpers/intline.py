from typing import Any
import numpy as np


def intline(x1: Any, x2: Any, y1: Any, y2: Any):
    """
    Integer-coordinate line drawing algorithm.
    MATLAB-faithful translation of intline.m.

    Parameters
    ----------
    x1, x2, y1, y2 : int-like
        Endpoint coordinates.

    Returns
    -------
    x, y : np.ndarray
        Integer coordinate vectors of the line segment (inclusive).
    """
    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Check for degenerate case.
    if dx == 0 and dy == 0:
        return np.asarray([x1], dtype=int), np.asarray([y1], dtype=int)

    flip = 0
    if dx >= dy:
        if x1 > x2:
            # Always "draw" from left to right.
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            flip = 1

        m = (y2 - y1) / (x2 - x1)
        x = np.arange(x1, x2 + 1, dtype=float)
        y = np.round(y1 + m * (x - x1))
    else:
        if y1 > y2:
            # Always "draw" from bottom to top.
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            flip = 1

        m = (x2 - x1) / (y2 - y1)
        y = np.arange(y1, y2 + 1, dtype=float)
        x = np.round(x1 + m * (y - y1))

    x = x.astype(int)
    y = y.astype(int)

    if flip:
        x = np.flipud(x)
        y = np.flipud(y)

    return x, y
