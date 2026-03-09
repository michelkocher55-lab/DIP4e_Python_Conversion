from typing import Any
import numpy as np


def multithresh3E(f: Any, T: Any):
    """
    Thresholds image f based on the thresholds in the array T.

    Parameters:
        f: Input image.
        T: Array of thresholds [T1, T2, ..., Tn]. Values must be in (0, 1).

    Returns:
        g: Thresholded image with n+1 levels.
    """
    T = np.array(T).flatten()

    if np.any(T <= 0) or np.any(T >= 1):
        raise ValueError("All values in array T must be > 0 and < 1")

    # Normalize f to [0, 1] if needed
    f = f.astype(float)
    max_val = f.max()
    if max_val > 1:
        f = f / max_val

    n = len(T)
    delta = 1.0 / n
    g = np.zeros_like(f)

    T_aug = np.concatenate([T, [1.0]])
    del_val = 0

    for k in range(n):
        del_val += delta
        lower = T_aug[k]
        upper = T_aug[k + 1]

        mask = (f > lower) & (f <= upper)
        g[mask] = del_val

    return g
