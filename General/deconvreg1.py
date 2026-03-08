import numpy as np
from scipy.optimize import fminbound
from skimage.util import img_as_float, img_as_ubyte, img_as_uint, img_as_int


_ALLOWED_IMAGE_DTYPES = {
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('int16'),
    np.dtype('float32'),
    np.dtype('float64'),
}


def _padlength(size_a, size_b, size_c):
    maxlen = max(len(size_a), len(size_b), len(size_c))

    def pad(s):
        s = list(s)
        return s + [1] * (maxlen - len(s))

    return pad(size_a), pad(size_b), pad(size_c)


def _psf2otf(psf, out_size):
    """ND psf2otf approximation compatible with MATLAB usage in deconvreg."""
    psf = np.asarray(psf, dtype=np.float64)
    out = np.zeros(out_size, dtype=np.float64)

    insert = tuple(slice(0, s) for s in psf.shape)
    out[insert] = psf

    # Circularly shift so PSF center is at [0,0,...]
    for ax, dim in enumerate(psf.shape):
        out = np.roll(out, -int(dim // 2), axis=ax)

    return np.fft.fftn(out)


def _default_regop_from_psf(size_psf):
    """
    Build default Laplacian REGOP based on non-singleton dimensions of PSF,
    then expand to PSF dimensionality by inserting singleton dimensions.
    """
    num_ns = [k for k, v in enumerate(size_psf) if v != 1]
    nsd = len(num_ns)

    if nsd == 0:
        # Degenerate fallback (should be rare): 1-D Laplacian
        reg_small = np.array([1.0, -2.0, 1.0], dtype=np.float64)
        nsd = 1
        num_ns = [0]
    elif nsd == 1:
        reg_small = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    else:
        shape = (3,) * nsd
        reg_small = np.zeros(shape, dtype=np.float64)
        center = [1] * nsd
        reg_small[tuple(center)] = -2.0 * nsd
        for d in range(nsd):
            idx_lo = center.copy()
            idx_hi = center.copy()
            idx_lo[d] = 0
            idx_hi[d] = 2
            reg_small[tuple(idx_lo)] = 1.0
            reg_small[tuple(idx_hi)] = 1.0

    # Expand to full PSF dimensionality with singleton insertion
    full_shape = [1] * len(size_psf)
    for d in num_ns:
        full_shape[d] = 3

    return reg_small.reshape(full_shape)


def _restore_class_like(original_dtype, arr):
    """Approximate MATLAB images.internal.changeClass behavior for common classes."""
    dt = np.dtype(original_dtype)

    if dt == np.dtype('float64'):
        return arr.astype(np.float64)
    if dt == np.dtype('float32'):
        return arr.astype(np.float32)
    if dt == np.dtype('uint8'):
        return img_as_ubyte(np.clip(arr, 0.0, 1.0))
    if dt == np.dtype('uint16'):
        return img_as_uint(np.clip(arr, 0.0, 1.0))
    if dt == np.dtype('int16'):
        return img_as_int(np.clip(arr, -1.0, 1.0))

    # Fallback (should not happen with validated inputs)
    return arr.astype(dt)


def _validate_real_finite(name, x):
    x = np.asarray(x)
    if not np.isrealobj(x):
        raise ValueError(f'{name} must be real.')
    if not np.all(np.isfinite(x)):
        raise ValueError(f'{name} must be finite.')


def deconvreg1(I, PSF, NP=0, LR=None, REGOP=None):
    """
    MATLAB-like DECONVREG (regularized deconvolution) with fminbound search.

    Parameters
    ----------
    I : ndarray
        Input image (uint8, uint16, int16, float32, float64).
    PSF : ndarray
        Point spread function (real, finite).
    NP : scalar, optional
        Additive noise power (default 0).
    LR : scalar or length-2 iterable, optional
        Lagrange multiplier or search range [Lmin, Lmax].
    REGOP : ndarray, optional
        Regularization operator. Default is Laplacian derived from PSF.

    Returns
    -------
    J : ndarray
        Restored image, same class as input I.
    LAGRA : float
        Selected Lagrange multiplier.
    """
    I = np.asarray(I)
    classI = I.dtype

    if classI not in _ALLOWED_IMAGE_DTYPES:
        raise ValueError('I must be uint8, uint16, int16, single, or double.')
    if I.size < 3:
        raise ValueError('Input image is too small.')
    _validate_real_finite('I', I)

    PSF = np.asarray(PSF, dtype=np.float64)
    _validate_real_finite('PSF', PSF)
    if PSF.size < 2:
        raise ValueError('PSF is too small.')
    if np.all(PSF == 0):
        raise ValueError('PSF cannot be all zeros.')

    if NP is None or (np.size(NP) == 0):
        NP = 0.0
    NP = float(np.asarray(NP).squeeze())

    if LR is None or (np.size(LR) == 0):
        LR = np.array([1e-9, 1e9], dtype=np.float64)
    else:
        LR = np.asarray(LR, dtype=np.float64).ravel()
        if LR.size > 2:
            raise ValueError('LRANGE must be scalar or 2 elements.')
        if LR.size == 2 and (LR[1] < LR[0]):
            raise ValueError('LRANGE must satisfy LR(2) >= LR(1).')

    if REGOP is None or (np.size(REGOP) == 0):
        REGOP = np.array([])
    else:
        REGOP = np.asarray(REGOP, dtype=np.float64)
        _validate_real_finite('REGOP', REGOP)

    # Validate dimensional compatibility (MATLAB-style checks)
    sizeI = list(I.shape)
    sizePSF = list(PSF.shape)
    sizeREG = list(REGOP.shape) if REGOP.size else [1]
    sizeI, sizePSF, sizeREG = _padlength(sizeI, sizePSF, sizeREG)

    num_ns_psf = [k for k, v in enumerate(sizePSF) if v != 1]
    if any(np.array(sizeI)[num_ns_psf] < np.array(sizePSF)[num_ns_psf]):
        raise ValueError('PSF dimensions must not exceed image dimensions.')

    if REGOP.size:
        num_ns_reg = [k for k, v in enumerate(sizeREG) if v != 1]
        if any(np.array(sizeI)[num_ns_reg] < np.array(sizeREG)[num_ns_reg]):
            raise ValueError('REGOP dimensions must not exceed image dimensions.')
        if any(np.array(sizePSF)[num_ns_reg] == 1):
            raise ValueError('Any non-singleton REGOP dimensions must correspond to non-singleton PSF dimensions.')

    # Convert image to processing float domain
    Iproc = img_as_float(I)

    # OTF of PSF
    otf = _psf2otf(PSF, sizeI)

    # Default REGOP if needed, then OTF
    if REGOP.size == 0:
        REGOP = _default_regop_from_psf(sizePSF)
    REGOP_otf = _psf2otf(REGOP, sizeI)

    fftnI = np.fft.fftn(Iproc)
    R2 = np.abs(REGOP_otf) ** 2
    H2 = np.abs(otf) ** 2

    # Determine LAGRA
    if LR.size == 1 or (LR.size == 2 and np.isclose(LR[0], LR[1])):
        LAGRA = float(LR[0])
    else:
        R4G2 = (R2 * np.abs(fftnI)) ** 2
        H4 = H2 ** 2
        R4 = R2 ** 2
        H2R22 = 2.0 * H2 * R2
        ScaledNP = NP * np.prod(sizeI)

        eps_sqrt = np.sqrt(np.finfo(np.float64).eps)

        def _res_offset(lagra):
            denom = H4 + lagra * H2R22 + (lagra ** 2) * R4 + eps_sqrt
            residuals = R4G2 / denom
            return abs((lagra ** 2) * np.sum(residuals) - ScaledNP)

        LAGRA = float(fminbound(_res_offset, LR[0], LR[1]))

    # Reconstruct
    Denom = H2 + LAGRA * R2
    Denom = np.maximum(Denom, np.sqrt(np.finfo(np.float64).eps))
    Nomin = np.conj(otf) * fftnI
    J = np.real(np.fft.ifftn(Nomin / Denom))

    J = _restore_class_like(classI, J)
    return J, LAGRA


# MATLAB-compatible function name alias

def deconvreg(*args, **kwargs):
    return deconvreg1(*args, **kwargs)
