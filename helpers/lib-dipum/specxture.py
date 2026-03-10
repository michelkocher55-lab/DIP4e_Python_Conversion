from typing import Any
import numpy as np

from helpers.intline import intline
from helpers.mat2gray import mat2gray


def halfcircle(r: Any, x0: Any, y0: Any):
    """
    Integer coordinates of a half circle of radius r centered at (x0, y0).
    Uses theta = 91..270 degrees (inclusive), matching MATLAB code.
    """
    theta = np.deg2rad(np.arange(91, 271, dtype=float))
    xc = np.round(r * np.cos(theta)).astype(int) + int(x0)
    yc = np.round(r * np.sin(theta)).astype(int) + int(y0)
    return xc, yc


def radial(x0: Any, y0: Any, x: Any, y: Any):
    """Integer coordinates of line segment from (x0,y0) to (x,y)."""
    xr, yr = intline(x0, x, y0, y)
    return xr, yr


def specxture(f: Any):
    """
    Computes spectral texture of an image.

    Parameters
    ----------
    f : ndarray
        Input image/region.

    Returns
    -------
    srad : ndarray
        Spectral energy distribution as a function of radius.
    sang : ndarray
        Spectral energy distribution as a function of angle (180 values).
    S : ndarray
        log(1 + centered spectrum), normalized to [0,1].
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 2:
        raise ValueError("f must be a 2D image.")

    # Centered magnitude spectrum.
    Smag = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    M, N = Smag.shape

    # Zero-frequency center in 0-based coordinates.
    x0 = M // 2
    y0 = N // 2

    # Maximum radius that keeps circle inside the spectrum bounds.
    rmax = int(np.floor(min(M, N) / 2.0 - 1))

    if rmax <= 0:
        srad = np.zeros(0, dtype=float)
        sang = np.zeros(180, dtype=float)
        S = mat2gray(np.log1p(Smag))
        return srad, sang, S

    # Radial distribution.
    srad = np.zeros(rmax, dtype=float)
    srad[0] = Smag[x0, y0]

    for r in range(2, rmax + 1):
        xc, yc = halfcircle(r, x0, y0)
        valid = (xc >= 0) & (xc < M) & (yc >= 0) & (yc < N)
        if np.any(valid):
            srad[r - 1] = np.sum(Smag[xc[valid], yc[valid]])

    # Angular distribution.
    xc, yc = halfcircle(rmax, x0, y0)
    sang = np.zeros(len(xc), dtype=float)

    for a in range(len(xc)):
        xr, yr = radial(x0, y0, xc[a], yc[a])
        valid = (xr >= 0) & (xr < M) & (yr >= 0) & (yr < N)
        if np.any(valid):
            sang[a] = np.sum(Smag[xr[valid], yr[valid]])

    # Log spectrum for display, scaled to [0,1].
    S = mat2gray(np.log1p(Smag))
    return srad, sang, S
