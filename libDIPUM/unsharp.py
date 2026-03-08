import numpy as np
from skimage.util import img_as_float
from scipy.ndimage import correlate


def _gaussian_kernel(n, sigma):
    """MATLAB-like fspecial('gaussian', n, sigma) kernel."""
    if isinstance(n, (tuple, list)):
        nr, nc = int(n[0]), int(n[1])
    else:
        nr = nc = int(n)

    # Coordinates centered around zero (matches MATLAB style).
    r = np.arange(-(nr - 1) / 2.0, (nr - 1) / 2.0 + 1.0)
    c = np.arange(-(nc - 1) / 2.0, (nc - 1) / 2.0 + 1.0)
    y, x = np.meshgrid(r, c, indexing='ij')

    w = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    s = np.sum(w)
    if s != 0:
        w = w / s
    return w


def unsharp(f, k, n, sigma):
    """
    UNSHARP Unsharp masking.

    [g, gb, gs] = unsharp(f, k, n, sigma)
    - f: input image
    - k: positive boosting parameter
    - n: gaussian kernel size (n x n)
    - sigma: gaussian std deviation

    Returns:
    - g: unsharp/highboost result
    - gb: blurred image
    - gs: sharp mask (f - gb)
    """
    f = img_as_float(f)
    w = _gaussian_kernel(n, sigma)

    # MATLAB: imfilter(f, w, 'replicate')
    gb = correlate(f, w, mode='nearest')
    gs = f - gb
    g = f + k * gs

    return g, gb, gs
