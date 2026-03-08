import numpy as np
from scipy.ndimage import convolve


def _mat2gray(x):
    """Scale input array to [0, 1], MATLAB mat2gray-style."""
    x = np.asarray(x, dtype=float)
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax > xmin:
        return (x - xmin) / (xmax - xmin)
    return np.zeros_like(x, dtype=float)


def colorgrad(f, T=None):
    """
    COLORGRAD Computes the vector gradient of an RGB image.

    [VG, A, PPG] = colorgrad(f, T)

    Parameters
    ----------
    f : ndarray
        RGB image of shape (M, N, 3).
    T : float, optional
        Threshold in [0, 1]. If provided, VG and PPG are thresholded
        logical arrays. If omitted, VG and PPG are scaled to [0, 1].

    Returns
    -------
    VG : ndarray
        Vector gradient magnitude (scaled or thresholded).
    A : ndarray
        Corresponding angle array (radians).
    PPG : ndarray
        Per-plane composite gradient (scaled or thresholded).
    """
    f = np.asarray(f)

    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError('Input image must be of size M-by-N-by-3')

    if T is not None and (T < 0 or T > 1):
        raise ValueError('T must be in the range [0 1]')

    # Sobel masks matching fspecial('sobel') and its transpose.
    sh = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    sv = sh.T

    # Compute x and y derivatives of RGB components (replicate boundaries).
    R = f[:, :, 0].astype(float)
    G = f[:, :, 1].astype(float)
    B = f[:, :, 2].astype(float)

    Rx = convolve(R, sh, mode='nearest')
    Ry = convolve(R, sv, mode='nearest')
    Gx = convolve(G, sh, mode='nearest')
    Gy = convolve(G, sv, mode='nearest')
    Bx = convolve(B, sh, mode='nearest')
    By = convolve(B, sv, mode='nearest')

    # Vector-gradient parameters.
    gxx = Rx**2 + Gx**2 + Bx**2
    gyy = Ry**2 + Gy**2 + By**2
    gxy = Rx * Ry + Gx * Gy + Bx * By

    A = 0.5 * np.arctan((2 * gxy) / (gxx - gyy + np.finfo(float).eps))
    G1 = 0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2 * A) + 2 * gxy * np.sin(2 * A))

    # Repeat for angle + pi/2.
    A = A + np.pi / 2
    G2 = 0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2 * A) + 2 * gxy * np.sin(2 * A))

    G1 = np.sqrt(np.maximum(G1, 0))
    G2 = np.sqrt(np.maximum(G2, 0))

    # Vector gradient magnitude.
    VG = _mat2gray(np.maximum(G1, G2))

    # Select corresponding angles.
    A[G2 > G1] = A[G2 > G1] + np.pi / 2

    # Per-plane composite gradient.
    RG = np.hypot(Rx, Ry)
    GG = np.hypot(Gx, Gy)
    BG = np.hypot(Bx, By)
    PPG = _mat2gray(RG + GG + BG)

    # Threshold if T provided.
    if T is not None:
        VG = VG > T
        PPG = PPG > T

    return VG, A, PPG
