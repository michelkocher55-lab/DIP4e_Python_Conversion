import numpy as np
from scipy.optimize import fminbound
from skimage.util import img_as_float


def _padlength(size_a, size_b, size_c):
    maxlen = max(len(size_a), len(size_b), len(size_c))
    def pad(s):
        return list(s) + [1] * (maxlen - len(s))
    return pad(size_a), pad(size_b), pad(size_c)


def _psf2otf(psf, out_size):
    """ND psf2otf matching MATLAB behavior."""
    psf = np.asarray(psf, dtype=float)
    out = np.zeros(out_size, dtype=float)

    insert_slices = tuple(slice(0, s) for s in psf.shape)
    out[insert_slices] = psf

    for axis, dim in enumerate(psf.shape):
        out = np.roll(out, -int(dim // 2), axis=axis)

    return np.fft.fftn(out)


def _default_regop(size_psf, num_ns_dim):
    nsd = len(num_ns_dim)
    if nsd == 1:
        regop = np.array([1, -2, 1], dtype=float)
    else:
        # Build ND Laplacian (3x3 core across NSD dims)
        base_shape = [3, 3] + [3] * (nsd - 2)
        regop = np.zeros(base_shape, dtype=float)

        for n in range(3, nsd + 1):
            idx = [2] * (n - 1) + [[1, 3]] + [2] * (nsd - n)
            regop[tuple(idx)] = 1

        idx = [slice(None), slice(None)] + [2] * (nsd - 2)
        regop[tuple(idx)] = np.array([[0, 1, 0], [1, -nsd * 2, 1], [0, 1, 0]], dtype=float)

    # Return the small Laplacian kernel; psf2otf will expand it to image size.
    return regop


def _change_class_like(original_dtype, arr):
    # Match MATLAB behavior for common integer types; otherwise return float.
    if original_dtype == np.uint8:
        return np.clip(arr, 0, 1) * 255.0
    if original_dtype == np.uint16:
        return np.clip(arr, 0, 1) * 65535.0
    return arr


def deconvreg(I, PSF, NP=0, LR=None, REGOP=None):
    """
    Deblur image using regularized filter (MATLAB deconvreg).

    Parameters:
        I: input image
        PSF: point-spread function
        NP: noise power (scalar)
        LR: lagrange range (scalar or [min, max])
        REGOP: regularization operator (optional)

    Returns:
        J: restored image
        LAGRA: Lagrange multiplier
    """
    if LR is None or (isinstance(LR, (list, tuple, np.ndarray)) and len(LR) == 0):
        LR = [1e-9, 1e9]

    I = np.asarray(I)
    classI = I.dtype
    I = img_as_float(I)

    sizeI = list(I.shape)
    sizePSF = list(np.asarray(PSF).shape)

    if np.prod(sizePSF) < 2:
        raise ValueError("PSF too small")
    if np.all(np.asarray(PSF) == 0):
        raise ValueError("PSF all zero")

    if NP is None:
        NP = 0
    if np.size(NP) > 1:
        raise ValueError("NP must be scalar")

    LR = np.asarray(LR, dtype=float).flatten()
    if LR.size > 2:
        raise ValueError("LRANGE size invalid")
    if LR.size == 0:
        LR = np.array([1e-9, 1e9], dtype=float)

    sizeREGOP = list(np.asarray(REGOP).shape) if REGOP is not None and np.size(REGOP) > 0 else [1]
    sizeI, sizePSF, sizeREGOP = _padlength(sizeI, sizePSF, sizeREGOP)

    numNSdim = [i for i, s in enumerate(sizePSF) if s != 1]

    if any(np.array(sizeI)[numNSdim] < np.array(sizePSF)[numNSdim]):
        raise ValueError("PSF too big")

    if REGOP is None or np.size(REGOP) == 0:
        REGOP = _default_regop(sizePSF, numNSdim)

    # OTF and REGOP in frequency domain
    otf = _psf2otf(PSF, sizeI)
    REGOP_otf = _psf2otf(REGOP, sizeI)

    fftnI = np.fft.fftn(I)
    R2 = np.abs(REGOP_otf) ** 2
    H2 = np.abs(otf) ** 2

    # LAGRA selection
    if LR.size == 1 or (LR.size == 2 and np.isclose(LR[0], LR[1])):
        LAGRA = float(LR[0])
    else:
        R4G2 = (R2 * np.abs(fftnI)) ** 2
        H4 = H2 ** 2
        R4 = R2 ** 2
        H2R22 = 2 * H2 * R2
        ScaledNP = float(NP) * np.prod(sizeI)

        def ResOffset(LAGRA_val):
            denom = H4 + LAGRA_val * H2R22 + (LAGRA_val ** 2) * R4 + np.sqrt(np.finfo(float).eps)
            residuals = R4G2 / denom
            return abs((LAGRA_val ** 2) * np.sum(residuals) - ScaledNP)

        LAGRA = float(fminbound(ResOffset, LR[0], LR[1]))

    Denom = H2 + LAGRA * R2
    Denom = np.maximum(Denom, np.sqrt(np.finfo(float).eps))
    Nomin = np.conj(otf) * fftnI

    J = np.real(np.fft.ifftn(Nomin / Denom))

    J = _change_class_like(classI, J)
    return J, LAGRA
