from typing import Any
import numpy as np


def ipercentile(h: Any, option: Any, V: Any):
    """
    Computes percentile of intensity histograms or intensity from percentile.

    Parameters:
    - h: 1D array-like, image intensity histogram.
    - option: 'percentile' or 'intensity'.
    - V: Float or integer.
        - If option is 'percentile', V is a percentile in [0, 1]. Returns intensity level.
        - If option is 'intensity', V is an intensity level (integer). Returns percentile in [0, 1].

    Returns:
    - Q: Intensity level or percentile, depending on option.
    """
    h = np.array(h, dtype=np.float64)
    # Normalize the histogram to unit area
    h = h / (np.sum(h) + np.finfo(float).eps)

    # Cumulative distribution
    C = np.cumsum(h)

    if option.lower() == "percentile":
        # V is a percentile; Q is an intensity level.
        if V < 0 or V > 1:
            raise ValueError('For "percentile", V must be in [0, 1].')
        elif V == 0:
            return 0
        elif V == 1:
            return len(h) - 1
        else:
            idx = np.searchsorted(C, V, side="left")
            return idx

    elif option.lower() == "intensity":
        # V is intensity; Q is percentile.
        if V < 0 or V > len(h) - 1:
            raise ValueError(f'For "intensity", V must be in [0, {len(h) - 1}].')

        if not np.isclose(V, int(V)):
            raise ValueError("V must be an integer.")

        V = int(V)

        if V == 0:
            return 0.0
        elif V == len(h) - 1:
            return 1.0
        else:
            return C[V]
    else:
        raise ValueError('Option must be "percentile" or "intensity".')
