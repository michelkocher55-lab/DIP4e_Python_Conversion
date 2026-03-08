import numpy as np


def lpc2mat(y, f=1):
    """
    Decompress a 1-D lossless predictive encoded matrix.
    """
    if f is None:
        f = 1

    # Reverse filter coefficients
    f = np.asarray(f, dtype=float).ravel()[::-1]

    y = np.asarray(y, dtype=float)
    m, n = y.shape
    order = len(f)

    # Duplicate filter for vectorizing
    f = np.tile(f, (m, 1))

    # Pad for first 'order' column decodes
    x = np.zeros((m, n + order), dtype=float)

    for j in range(n):
        jj = j + order
        x[:, jj] = y[:, j] + np.round(np.sum(
            f[:, order-1::-1] * x[:, (jj - 1):(jj - order - 1):-1], axis=1
        ))

    # Remove left padding
    x = x[:, order:]
    return x
