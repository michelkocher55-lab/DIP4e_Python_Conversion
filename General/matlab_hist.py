from typing import Any
import numpy as np


def matlab_hist(x: Any, centers: Any):
    """
    Replicates MATLAB hist(x, centers) behavior.

    Parameters
    ----------
    x : ndarray
        Input data (any shape)
    centers : ndarray
        Bin centers (1-D)

    Returns
    -------
    counts : ndarray
        Histogram counts, same length as centers
    """
    x = np.asarray(x).ravel()
    centers = np.asarray(centers)

    # Compute bin edges halfway between centers
    edges = np.empty(len(centers) + 1, dtype=centers.dtype)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Digitize and count
    idx = np.digitize(x, edges) - 1
    counts = np.bincount(idx, minlength=len(centers))

    return counts
