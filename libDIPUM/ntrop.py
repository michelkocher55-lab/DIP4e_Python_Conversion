import numpy as np


def ntrop(x, n=256):
    """
    First-order estimate of the entropy of a matrix.
    H = ntrop(x, n) where n is number of symbols (bins).
    """
    x = np.asarray(x, dtype=float)
    # Histogram with n bins
    xh, _ = np.histogram(x.ravel(), bins=int(n))
    xh = xh.astype(float)
    xh = xh / np.sum(xh)
    # Mask zeros to avoid log2(0)
    nz = xh > 0
    h = -np.sum(xh[nz] * np.log2(xh[nz]))
    return h
