import numpy as np


def psf2otf(psf, out_size):
    """
    Convert a point-spread function (PSF) to an optical transfer function (OTF).
    Matches MATLAB psf2otf behavior for 2D inputs.
    """
    psf = np.asarray(psf, dtype=float)
    out = np.zeros(out_size, dtype=float)

    psf_shape = psf.shape
    out[:psf_shape[0], :psf_shape[1]] = psf

    # Circularly shift so that the PSF center is at the (0,0) element
    for axis, dim in enumerate(psf_shape):
        out = np.roll(out, -int(dim // 2), axis=axis)

    return np.fft.fft2(out)


def deconvwnr(g, psf, nsr):
    """
    Wiener deconvolution (frequency-domain). Equivalent to MATLAB deconvwnr.

    Parameters:
        g: degraded image (spatial domain)
        psf: point-spread function (spatial domain)
        nsr: noise-to-signal power ratio (scalar)

    Returns:
        f_hat: restored image
    """
    g = np.asarray(g)
    G = np.fft.fft2(g)

    H = psf2otf(psf, g.shape)

    nsr_val = nsr
    if np.isscalar(nsr_val):
        denom = (np.abs(H) ** 2) + nsr_val
    else:
        denom = (np.abs(H) ** 2) + np.asarray(nsr_val)

    F_hat = (np.conj(H) / denom) * G
    f_hat = np.fft.ifft2(F_hat)

    return np.real(f_hat)
