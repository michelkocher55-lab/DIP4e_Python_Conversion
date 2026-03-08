import numpy as np
from scipy.ndimage import rotate

try:
    from skimage.transform import radon as _skimage_radon
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def _default_r_len(shape):
    m, n = shape
    # MATLAB doc: 2*ceil(norm(size(I)-floor((size(I)-1)/2)-1))+3
    size_i = np.array([m, n], dtype=float)
    floor_term = np.floor((size_i - 1) / 2.0)
    val = size_i - floor_term - 1
    return int(2 * np.ceil(np.linalg.norm(val)) + 3)


def radon(I, theta=None, n=None, use_skimage=True):
    """
    Radon transform (MATLAB-like).

    Parameters:
        I: 2D image (numeric)
        theta: angles in degrees (vector). Default 0:179
        n: optional number of projection samples (grandfathered syntax)

    Returns:
        P: Radon transform (rows: r, cols: theta)
        r: radial coordinates
    """
    I = np.asarray(I)
    if I.ndim != 2:
        raise ValueError("I must be a 2D array")

    if theta is None:
        theta = np.arange(0, 180)
    theta = np.asarray(theta, dtype=float).ravel()

    # Convert image to float
    I = I.astype(float, copy=False)

    # If available, use skimage's radon (closer to MATLAB behavior/geometry)
    if use_skimage and _HAS_SKIMAGE:
        P = _skimage_radon(I, theta=theta, circle=False)
        r = np.arange(P.shape[0], dtype=float) - (P.shape[0] - 1) / 2.0
        if n is not None:
            n = int(n)
            if n <= 0:
                raise ValueError("n must be positive")
            if P.shape[0] != n:
                new_r = np.linspace(r.min(), r.max(), n)
                P = np.vstack([np.interp(new_r, r, P[:, i]) for i in range(P.shape[1])]).T
                P = P * (len(r) / len(new_r))
                r = new_r
        return P, r

    # Determine projection length and pad image to square
    proj_len = _default_r_len(I.shape)
    pad_m = max(0, proj_len - I.shape[0])
    pad_n = max(0, proj_len - I.shape[1])
    pad_before_m = pad_m // 2
    pad_after_m = pad_m - pad_before_m
    pad_before_n = pad_n // 2
    pad_after_n = pad_n - pad_before_n

    Ipad = np.pad(I, ((pad_before_m, pad_after_m), (pad_before_n, pad_after_n)),
                 mode='constant', constant_values=0)

    # Ensure square
    if Ipad.shape[0] != Ipad.shape[1]:
        size = max(Ipad.shape)
        extra_m = size - Ipad.shape[0]
        extra_n = size - Ipad.shape[1]
        Ipad = np.pad(Ipad,
                      ((extra_m // 2, extra_m - extra_m // 2),
                       (extra_n // 2, extra_n - extra_n // 2)),
                      mode='constant', constant_values=0)

    # Radial coordinates
    r = np.arange(Ipad.shape[0], dtype=float) - (Ipad.shape[0] - 1) / 2.0

    # Compute projections
    P = np.zeros((Ipad.shape[0], theta.size), dtype=float)
    for i, ang in enumerate(theta):
        # Rotate so projection direction aligns with columns
        rot = rotate(Ipad, angle=-ang, reshape=False, order=1, mode='constant', cval=0.0)
        P[:, i] = np.sum(rot, axis=0)

    # Optional resizing to n samples
    if n is not None:
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        if P.shape[0] != n:
            new_r = np.linspace(r.min(), r.max(), n)
            P = np.vstack([np.interp(new_r, r, P[:, i]) for i in range(P.shape[1])]).T
            P = P * (len(r) / len(new_r))
            r = new_r

    return P, r
