import numpy as np


def zoneplate(K, del_, R=None):
    """
    ZONEPLATE generates a zoneplate image.

    g = zoneplate(K, del_, R=None)

    Parameters
    ----------
    K : float
        Extent in both axes. Grid is approximately from -K to K.
    del_ : float
        Sampling interval.
    R : float or None, optional
        If provided, pixels outside the circular zone plate are set to R.
        R must be in [0, 1].

    Returns
    -------
    g : ndarray
        Zone plate image in floating-point range [0, 1].
    """
    if del_ <= 0:
        raise ValueError('del must be positive.')

    # MATLAB: -K:del:K (inclusive when possible)
    t = np.arange(-K, K + del_ / 2.0, del_, dtype=np.float64)
    x, y = np.meshgrid(t, t)

    # Zone plate formula
    g = (1.0 + np.cos(x * x + y * y)) / 2.0

    # Optional circular border set to R
    if R is not None:
        if R < 0 or R > 1:
            raise ValueError('Values of R must be in the range [0,1].')

        M = g.shape[0]
        center = (M - 1) / 2.0
        radius = (M - 1) / 2.0

        yy, xx = np.indices((M, M), dtype=np.float64)
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        g[dist > (radius - 1.0)] = R

    return g
