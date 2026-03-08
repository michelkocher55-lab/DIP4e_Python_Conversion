import numpy as np


def frdescp(s):
    """
    Computes Fourier descriptors.

    Parameters
    ----------
    s : array_like, shape (np, 2)
        Ordered boundary coordinates.

    Returns
    -------
    z : ndarray, shape (np,)
        Complex Fourier descriptors.

    Notes
    -----
    If np is odd, the last point is duplicated so output length is even.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError('S must be of size np-by-2.')

    np_ = s.shape[0]
    if np_ % 2 != 0:
        s = np.vstack((s, s[-1, :]))
        np_ += 1

    # Alternating +1/-1 sequence for transform centering.
    x = np.arange(np_)
    m = ((-1) ** x).astype(float)

    s_centered = s.copy()
    s_centered[:, 0] = m * s_centered[:, 0]
    s_centered[:, 1] = m * s_centered[:, 1]

    # Convert coordinates to complex sequence and compute descriptors.
    sc = s_centered[:, 0] + 1j * s_centered[:, 1]
    z = np.fft.fft(sc)
    return z
