"""Render SIFT keypoints overlayed on an image.

MATLAB-like signature:
    out = showkeys(image, locs)
"""

from __future__ import annotations
from typing import Any

import numpy as np
import imageio.v2 as iio


def _transform_line_points(keypoint: Any, x1: Any, y1: Any, x2: Any, y2: Any):
    """Return transformed line endpoints using keypoint [row, col, scale, orientation]."""
    # Approximate descriptor support radius used in original MATLAB code.
    length = 6.0 * float(keypoint[2])

    s = np.sin(float(keypoint[3]))
    c = np.cos(float(keypoint[3]))

    r1 = float(keypoint[0]) - length * (c * y1 + s * x1)
    c1 = float(keypoint[1]) + length * (-s * y1 + c * x1)
    r2 = float(keypoint[0]) - length * (c * y2 + s * x2)
    c2 = float(keypoint[1]) + length * (-s * y2 + c * x2)
    return r1, c1, r2, c2


def _draw_line_rgb(
    out: np.ndarray, r1: float, c1: float, r2: float, c2: float, color: Any
) -> None:
    """_draw_line_rgb."""
    rows, cols = out.shape[:2]
    n = int(max(abs(r2 - r1), abs(c2 - c1))) + 1
    n = max(n, 2)
    rr = np.rint(np.linspace(r1, r2, n)).astype(np.int64)
    cc = np.rint(np.linspace(c1, c2, n)).astype(np.int64)
    valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    rr = rr[valid]
    cc = cc[valid]
    if rr.size == 0:
        return

    # Slightly thicken lines for better visibility.
    thick = 1
    for dr in range(-thick, thick + 1):
        for dc in range(-thick, thick + 1):
            rr2 = rr + dr
            cc2 = cc + dc
            valid2 = (rr2 >= 0) & (rr2 < rows) & (cc2 >= 0) & (cc2 < cols)
            if np.any(valid2):
                out[rr2[valid2], cc2[valid2], :] = color


def _to_float_gray(image: np.ndarray) -> np.ndarray:
    """_to_float_gray."""
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float64, copy=False)
    if img.size == 0:
        return img
    if np.min(img) < 0:
        img = img - np.min(img)
    mx = np.max(img)
    if mx > 1.0:
        img = img / mx
    return np.clip(img, 0.0, 1.0)


def showkeys(image: Any, locs: Any):
    """Render image with SIFT keypoints and return the overlay image.

    Parameters
    ----------
    image : ndarray or str
        Grayscale image array, or an image filename.
    locs : ndarray, shape (K, 4)
        Keypoint rows as [row, col, scale, orientation].

    Returns
    -------
    out : ndarray, shape (rows, cols, 3), float64
        RGB image with yellow keypoint arrows overlayed.
    """
    print("Rendering SIFT keypoints ...")

    if isinstance(image, (str, bytes)):
        image = iio.imread(image)
    image = _to_float_gray(image)

    locs = np.asarray(locs, dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] != 4:
        raise ValueError("locs must be a Kx4 array: [row, col, scale, orientation].")

    rows, cols = image.shape[:2]
    out = np.repeat(image[:, :, None], 3, axis=2)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float64)

    for i in range(locs.shape[0]):
        key = locs[i, :]
        r1, c1, r2, c2 = _transform_line_points(key, 0.0, 0.0, 1.0, 0.0)
        _draw_line_rgb(out, r1, c1, r2, c2, yellow)
        r1, c1, r2, c2 = _transform_line_points(key, 0.85, 0.1, 1.0, 0.0)
        _draw_line_rgb(out, r1, c1, r2, c2, yellow)
        r1, c1, r2, c2 = _transform_line_points(key, 0.85, -0.1, 1.0, 0.0)
        _draw_line_rgb(out, r1, c1, r2, c2, yellow)

    return out


__all__ = ["showkeys"]
