import numpy as np

try:
    from dftuv import dftuv
except ImportError:
    try:
        from .dftuv import dftuv
    except ImportError:
        pass


def homomorphictf(P, Q, gL, gH, c, D0):
    """
    Homomorphic filter transfer function.

    Implements (Gonzalez and Woods, 4th ed.):
        H(u,v) = (gH - gL) * (1 - exp(-c * (D^2(u,v) / D0^2))) + gL

    where D(u,v) is the distance from the center of the P-by-Q frequency
    rectangle. The returned transfer function is uncentered (origin at [0,0]),
    matching DIP convention. Use fftshift to center for display if needed.

    Parameters
    ----------
    P, Q : int
        Size of the frequency-domain filter.
    gL, gH : float
        Low- and high-frequency gains. Typically gH > gL.
    c : float
        Sharpness constant.
    D0 : float
        Cutoff frequency.

    Returns
    -------
    H : ndarray
        Uncentered homomorphic transfer function of shape (P, Q).
    """
    if D0 == 0:
        raise ValueError("D0 must be non-zero.")

    U, V = dftuv(P, Q)
    Dsq = U**2 + V**2

    H = (gH - gL) * (1.0 - np.exp(-c * (Dsq / (D0**2)))) + gL
    return H
