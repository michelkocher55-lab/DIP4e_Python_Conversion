import numpy as np


def lpc2mat2d(y, alpha, beta, gamma):
    """
    Decode 2-D lossless predictive coding.
    fHat(x,y) = alpha*f(x, y-1) + beta*f(x-1, y) + gamma*f(x-1, y-1)
    """
    y = np.asarray(y, dtype=float)
    m, n = y.shape
    f = np.zeros((m, n), dtype=float)

    for i in range(m):
        for j in range(n):
            left = f[i, j - 1] if j - 1 >= 0 else 0.0
            up = f[i - 1, j] if i - 1 >= 0 else 0.0
            upleft = f[i - 1, j - 1] if (i - 1 >= 0 and j - 1 >= 0) else 0.0
            fhat = alpha * left + beta * up + gamma * upleft
            f[i, j] = y[i, j] + np.round(fhat)

    return f
