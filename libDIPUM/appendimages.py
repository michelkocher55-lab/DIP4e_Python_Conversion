"""Append two images side-by-side (MATLAB appendimages equivalent)."""

from __future__ import annotations

import numpy as np


def appendimages(image1, image2):
    """Return a new image that appends the two images side-by-side."""
    im1 = np.asarray(image1)
    im2 = np.asarray(image2)

    if im1.ndim != 2 or im2.ndim != 2:
        raise ValueError("appendimages expects two 2-D grayscale images.")

    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    cols1 = im1.shape[1]
    cols2 = im2.shape[1]

    # Match MATLAB behavior: extend the shorter image with zero rows.
    if rows1 < rows2:
        out1 = np.zeros((rows2, cols1), dtype=np.result_type(im1.dtype, im2.dtype))
        out1[:rows1, :] = im1
        out2 = im2.astype(out1.dtype, copy=False)
    elif rows2 < rows1:
        out2 = np.zeros((rows1, cols2), dtype=np.result_type(im1.dtype, im2.dtype))
        out2[:rows2, :] = im2
        out1 = im1.astype(out2.dtype, copy=False)
    else:
        dt = np.result_type(im1.dtype, im2.dtype)
        out1 = im1.astype(dt, copy=False)
        out2 = im2.astype(dt, copy=False)

    return np.hstack((out1, out2))


__all__ = ["appendimages"]

