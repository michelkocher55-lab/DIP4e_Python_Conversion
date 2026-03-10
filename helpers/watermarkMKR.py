"""Watermark insertion/attack/detection (MATLAB watermarkMKR equivalent)."""

from __future__ import annotations
from typing import Any

import os
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage import rotate
from skimage.exposure import equalize_hist
from skimage.io import imread
from PIL import Image

from helpers.wiener2 import wiener2
from helpers.imnoise2 import imnoise2
from helpers.mat2gray import mat2gray


def _dct2(a: np.ndarray) -> np.ndarray:
    """_dct2."""
    return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(a: np.ndarray) -> np.ndarray:
    """_idct2."""
    return idct(idct(a, axis=1, norm="ortho"), axis=0, norm="ortho")


def _corr2(a: np.ndarray, b: np.ndarray) -> float:
    """_corr2."""
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    if aa.size != bb.size:
        raise ValueError("Inputs for corr2 must have the same number of elements.")
    if aa.size < 2:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _as_uint8(img: np.ndarray) -> np.ndarray:
    """_as_uint8."""
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating):
        if x.max() <= 1.0:
            x = x * 255.0
        return np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """_jpeg_roundtrip."""
    img_u8 = _as_uint8(img)
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tmpname = tf.name
    try:
        Image.fromarray(img_u8).save(tmpname, format="JPEG", quality=quality)
        out = np.asarray(Image.open(tmpname))
    finally:
        if os.path.exists(tmpname):
            os.remove(tmpname)
    return _as_uint8(out)


def watermarkMKR(
    f: Any, w: Any, alpha: Any, attack: Any = "none", compare: Any = "same"
):
    """Insert, attack, and detect a pseudorandom watermark.

    Parameters
    ----------
    f : ndarray
        Grayscale input image.
    w : ndarray
        Watermark sequence (typically length 1000).
    alpha : float
        Insertion strength.
    attack : str
        One of: 'none', 'gaussian'/'noise', 'filter', 'jpeg10', 'jpeg70',
        'nomark', 'difmark', 'heq', 'rotate'.
    compare : str
        'same' or 'largest'.

    Returns
    -------
    wi : ndarray
        Watermarked/attacked image.
    di : ndarray
        Difference image (uint8) between original and pre-attack watermarked image.
    c : float
        Correlation coefficient between original and extracted marks.
    """
    f = np.asarray(f)
    if f.ndim != 2:
        raise ValueError("f must be a 2-D grayscale image.")

    w = np.asarray(w, dtype=float).reshape(-1, 1)
    NW = w.shape[0]

    # DCT2
    tfrm = _dct2(np.asarray(f, dtype=float))

    # Highest |DCT| values and indices
    flat = tfrm.ravel()
    order = np.argsort(np.abs(flat))[::-1]
    index = order[:NW]
    coef = flat[index].copy()  # keep sign

    # Insert watermark: c' = c * (1 + alpha * w)
    flat[index] = coef * (1.0 + alpha * w.ravel())
    tfrm_mod = flat.reshape(tfrm.shape)
    wi = _as_uint8(_idct2(tfrm_mod))

    # Difference image in spatial domain (before attacks)
    d = np.asarray(f, dtype=float) - wi.astype(float)
    di = _as_uint8(255.0 * mat2gray(d))

    # Simulate attacks
    attack = str(attack).lower()
    if attack == "none":
        pass
    elif attack == "jpeg10":
        wi = _jpeg_roundtrip(wi, quality=10)
    elif attack == "jpeg70":
        wi = _jpeg_roundtrip(wi, quality=70)
    elif attack == "filter":
        wi, _ = wiener2(wi, [7, 7])
    elif attack in ("noise", "gaussian"):
        wi, _ = imnoise2(wi, "gaussian", 0, np.sqrt(0.005))
    elif attack == "nomark":
        wi = imread("tracy.tif")
    elif attack == "difmark":
        wi = imread("wlena_2.tif")
    elif attack == "heq":
        wi = _as_uint8(equalize_hist(_as_uint8(wi)) * 255.0)
    elif attack == "rotate":
        # MATLAB equivalent: imrotate(..., 'bilinear', 'crop') twice.
        # Use constant zero fill (not nearest) and recast to uint8 between
        # rotations to mimic MATLAB class-preserving behavior.
        wi = rotate(
            _as_uint8(wi),
            1.0,
            reshape=False,
            order=1,  # bilinear
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        wi = _as_uint8(wi)
        wi = rotate(
            wi,
            -1.0,
            reshape=False,
            order=1,  # bilinear
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        wi = _as_uint8(wi)
    else:
        raise ValueError(f"Unknown attack: {attack}")

    # Extract watermark from attacked image
    tst = _dct2(np.asarray(wi, dtype=float))

    if compare == "same":
        wtst = tst.ravel()[index]
    elif compare == "largest":
        # MATLAB code uses 1000 largest coefficients.
        K = min(1000, tst.size)
        idx2 = np.argsort(np.abs(tst.ravel()))[::-1][:K]
        wtst = tst.ravel()[idx2]
        if K != NW:
            raise ValueError(
                f"compare='largest' extracted {K} coeffs but watermark length is {NW}."
            )
    else:
        raise ValueError("compare must be 'same' or 'largest'.")

    wtst = (wtst - coef) / (alpha * coef)

    # Correlation coefficient
    c = _corr2(w.ravel(), np.asarray(wtst).ravel())

    return wi, di, c


__all__ = ["watermarkMKR"]
