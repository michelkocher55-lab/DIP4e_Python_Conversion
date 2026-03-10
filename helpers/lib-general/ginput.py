from typing import Any
import matplotlib.pyplot as plt
import numpy as np


def ginput(n: Any = 1, timeout: Any = 30, show_clicks: Any = True):
    """
    Graphical input from mouse clicks.

    Parameters:
        n: Number of clicks to accumulate. If negative, accumulate until <Enter>.
        timeout: Timeout in seconds.
        show_clicks: Whether to show a crosshair/marker at click locations.

    Returns:
        x, y: Arrays of coordinates.
    """
    # Simply wrap matplotlib ginput
    # It returns list of tuples [(x1,y1), (x2,y2), ...]
    # MATLAB returns [x, y] vectors.
    try:
        pts = plt.ginput(n, timeout=timeout, show_clicks=show_clicks)
    except Exception as e:
        print(f"Error in ginput: {e}")
        return np.array([]), np.array([])

    if not pts:
        return np.array([]), np.array([])

    pts = np.array(pts)
    x = pts[:, 0]
    y = pts[:, 1]

    return x, y
