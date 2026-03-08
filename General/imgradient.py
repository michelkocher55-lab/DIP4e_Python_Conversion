import numpy as np
from scipy.ndimage import correlate


def _is_method(x):
    return isinstance(x, (str, np.str_))


def _canonical_method(method):
    m = str(method).strip().lower()
    aliases = {
        "sobel": "sobel",
        "prewitt": "prewitt",
        "roberts": "roberts",
        "central": "centraldifference",
        "centraldifference": "centraldifference",
        "intermediate": "intermediatedifference",
        "intermediatedifference": "intermediatedifference",
    }
    if m not in aliases:
        raise ValueError(f"Unknown METHOD: {method}")
    return aliases[m]


def _validate_2d_real(name, arr):
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix.")
    if np.iscomplexobj(a):
        raise ValueError(f"{name} must be real.")
    return a


def _imgradientxy(I, method):
    """
    Directional gradients with replicate boundary handling.
    """
    I = np.asarray(I)
    method = _canonical_method(method)

    if method == "sobel":
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=I.dtype) / 8.0
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=I.dtype) / 8.0
        Gx = correlate(I, kx, mode="nearest")
        Gy = correlate(I, ky, mode="nearest")

    elif method == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=I.dtype) / 6.0
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=I.dtype) / 6.0
        Gx = correlate(I, kx, mode="nearest")
        Gy = correlate(I, ky, mode="nearest")

    elif method == "centraldifference":
        # dI/dx = (I(x+1) - I(x-1))/2, dI/dy = (I(y+1) - I(y-1))/2
        kx = np.array([[-0.5, 0.0, 0.5]], dtype=I.dtype)
        ky = np.array([[0.5], [0.0], [-0.5]], dtype=I.dtype)
        Gx = correlate(I, kx, mode="nearest")
        Gy = correlate(I, ky, mode="nearest")

    elif method == "intermediatedifference":
        # dI/dx = I(x+1) - I(x), dI/dy = I(y+1) - I(y)
        Gx = np.empty_like(I)
        Gy = np.empty_like(I)

        Gx[:, :-1] = I[:, 1:] - I[:, :-1]
        Gx[:, -1] = 0

        Gy[:-1, :] = I[1:, :] - I[:-1, :]
        Gy[-1, :] = 0

    else:
        raise ValueError(f"Unsupported METHOD: {method}")

    return Gx, Gy


def imgradient(*args):
    """
    MATLAB-like IMGRADIENT.

    Usage:
      Gmag, Gdir = imgradient(I)
      Gmag, Gdir = imgradient(I, method)
      Gmag, Gdir = imgradient(Gx, Gy)
    """
    if len(args) < 1 or len(args) > 2:
        raise ValueError("imgradient expects 1 or 2 input arguments.")

    I = None
    Gx = None
    Gy = None
    method = "sobel"

    if len(args) == 1:
        I = _validate_2d_real("I", args[0])
    else:
        if _is_method(args[1]):
            I = _validate_2d_real("I", args[0])
            method = _canonical_method(args[1])
        else:
            Gx = _validate_2d_real("Gx", args[0])
            Gy = _validate_2d_real("Gy", args[1])
            if Gx.shape != Gy.shape:
                raise ValueError("Gx and Gy must be the same size.")

    # Class support: keep single output when single input is involved.
    want_single = False
    if I is not None:
        want_single = np.asarray(I).dtype == np.float32
    else:
        want_single = (np.asarray(Gx).dtype == np.float32) or (np.asarray(Gy).dtype == np.float32)

    work_dtype = np.float32 if want_single else np.float64

    if I is not None:
        Iw = np.asarray(I, dtype=work_dtype)
        if method == "roberts":
            Gx = correlate(Iw, np.array([[1, 0], [0, -1]], dtype=work_dtype), mode="nearest")
            Gy = correlate(Iw, np.array([[0, 1], [-1, 0]], dtype=work_dtype), mode="nearest")
        else:
            Gx, Gy = _imgradientxy(Iw, method)
    else:
        Gx = np.asarray(Gx, dtype=work_dtype)
        Gy = np.asarray(Gy, dtype=work_dtype)

    Gmag = np.hypot(Gx, Gy).astype(work_dtype, copy=False)

    if method == "roberts":
        Gdir = np.zeros_like(Gx, dtype=work_dtype)
        nz = ~((Gx == 0) & (Gy == 0))
        Gdir[nz] = np.arctan2(Gy[nz], -Gx[nz]) - (np.pi / 4.0)
        low = Gdir < -np.pi
        Gdir[low] = Gdir[low] + 2.0 * np.pi
        Gdir = (Gdir * (180.0 / np.pi)).astype(work_dtype, copy=False)
    else:
        Gdir = (np.arctan2(-Gy, Gx) * (180.0 / np.pi)).astype(work_dtype, copy=False)

    return Gmag, Gdir
