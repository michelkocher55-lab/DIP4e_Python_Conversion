import numpy as np


def mat2lpc(x, f=1):
    """
    Compress a matrix using 1-D lossless predictive coding.
    """
    if f is None:
        f = 1

    x = np.asarray(x, dtype=float)
    m, n = x.shape

    # Ensure filter is iterable
    if np.isscalar(f):
        f = [f]
    f = np.asarray(f, dtype=float).ravel()

    p = np.zeros((m, n), dtype=float)
    xs = x.copy()
    zc = np.zeros((m, 1), dtype=float)

    for coeff in f:
        xs = np.hstack([zc, xs[:, :-1]])
        p = p + coeff * xs

    y = x - np.round(p)
    return y
