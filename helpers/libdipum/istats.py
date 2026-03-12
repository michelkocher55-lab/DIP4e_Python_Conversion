from typing import Any
import numpy as np


def istats(i: Any):
    """
    ISTATS computes mean, variance, and standard deviation of a matrix.

    Parameters
    ----------
    i : array-like

    Returns
    -------
    m : float
        Mean of all elements.
    v : float
        Variance of all elements.
    s : float
        Standard deviation of all elements.
    """
    i = np.asarray(i, dtype=float)
    m = float(np.mean(i))
    s = float(np.std(i, ddof=1))
    v = float(np.var(i, ddof=1))
    return m, v, s
