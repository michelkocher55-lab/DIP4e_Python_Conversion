import numpy as np
from skimage.measure import find_contours

from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e


def _zero_level_contours_all(phi):
    """Return all zero-level contours as 2xN [x; y] with NaN separators."""
    contours = find_contours(phi, level=0.0)
    if len(contours) == 0:
        return np.zeros((2, 0), dtype=float)

    xs = []
    ys = []
    for cc in contours:
        # find_contours gives (row, col); keep contourc-like convention row0=x(col), row1=y(row)
        xs.append(cc[:, 1])
        ys.append(cc[:, 0])
        xs.append(np.array([np.nan]))
        ys.append(np.array([np.nan]))

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    return np.vstack([x_all, y_all])


def LevelSetRegionBased(f, binmask, mu, nu, lambda1, lambda2, niter):
    """
    Region-based level-set segmentation.

    Transcoding of MATLAB:
      c = LevelSetRegionBased(f, binmask, mu, nu, lambda1, lambda2, niter)
    """
    # Create initial level set function from the mask.
    phi = levelSetFunction4e('mask', binmask)

    for I in range(1, int(niter) + 1):
        F = levelSetForce4e('regioncurve', [f, phi, mu, nu, lambda1, lambda2], ['Fn', 'Cn'])
        phi = levelSetIterate4e(phi, F)

        # Update every 5 iterations.
        if I % 5 == 0:
            phi = levelSetReInit4e(phi, 5, 0.5)

    c = _zero_level_contours_all(phi)
    return c
