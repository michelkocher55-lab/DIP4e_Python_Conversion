import numpy as np


def ifrdescp(z, nd=None):
    """
    Computes inverse Fourier descriptors.

    Parameters
    ----------
    z : array_like
        Sequence of Fourier descriptors (must have even length).
    nd : int, optional
        Number of descriptors used in the inverse; must be even and <= len(z).
        Defaults to len(z).

    Returns
    -------
    s : ndarray, shape (len(z), 2)
        Boundary coordinates [x, y] as columns.
    """
    z = np.asarray(z).reshape(-1).astype(complex)
    np_ = z.size

    if nd is None:
        nd = np_

    if np_ % 2 != 0:
        raise ValueError('length(z) must be an even integer.')
    if int(nd) != nd or nd % 2 != 0:
        raise ValueError('nd must be an even integer.')
    nd = int(nd)
    if nd < 0 or nd > np_:
        raise ValueError('nd must satisfy 0 <= nd <= length(z).')

    # Alternating sequence of 1 and -1 (undo centering done in frdescp).
    x = np.arange(np_)
    m = ((-1) ** x).astype(float)

    # Use only nd descriptors in the inverse.
    z_use = z.copy()
    d = (np_ - nd) // 2
    if d > 0:
        z_use[:d] = 0
        z_use[np_ - d:] = 0

    # Inverse transform and boundary coordinates.
    zz = np.fft.ifft(z_use)
    s = np.zeros((np_, 2), dtype=float)
    s[:, 0] = np.real(zz)
    s[:, 1] = np.imag(zz)

    # Undo centering.
    s[:, 0] = m * s[:, 0]
    s[:, 1] = m * s[:, 1]

    return s
