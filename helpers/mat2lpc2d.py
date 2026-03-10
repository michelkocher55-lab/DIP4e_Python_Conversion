from typing import Any
import numpy as np


def mat2lpc2d(f: Any, alpha: Any, beta: Any, gamma: Any):
    """
    2-D lossless predictive coding.
    fHat(x,y) = alpha*f(x, y-1) + beta*f(x-1, y) + gamma*f(x-1, y-1)
    """
    f = np.asarray(f, dtype=float)
    m, n = f.shape
    fhat = np.zeros((m, n), dtype=float)

    # zero padding on top/left is implicit by index checks
    for i in range(m):
        for j in range(n):
            left = f[i, j - 1] if j - 1 >= 0 else 0.0
            up = f[i - 1, j] if i - 1 >= 0 else 0.0
            upleft = f[i - 1, j - 1] if (i - 1 >= 0 and j - 1 >= 0) else 0.0
            fhat[i, j] = alpha * left + beta * up + gamma * upleft

    y = f - np.round(fhat)
    return y
