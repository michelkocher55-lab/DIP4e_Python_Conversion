"""Match SIFT keypoints between two images (MATLAB-style)."""

from __future__ import annotations

import numpy as np

from libDIPUM.appendimages import appendimages
from libDIPUM.sift import sift


def _to_float_gray(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image).astype(np.float64, copy=False)
    if img.size == 0:
        return img
    if np.min(img) < 0:
        img = img - np.min(img)
    mx = np.max(img)
    if mx > 1.0:
        img = img / mx
    return np.clip(img, 0.0, 1.0)


def _draw_line_rgb(out: np.ndarray, r1: float, c1: float, r2: float, c2: float, color) -> None:
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


def match(image1, image2):
    """Find SIFT matches and return match count + overlay image.

    Parameters
    ----------
    image1, image2 : str
        Image filenames.

    Returns
    -------
    num : int
        Number of accepted matches.
    overlay : ndarray, shape (rows, cols, 3), float64
        RGB image with yellow lines joining matched keypoints.
    """
    im1, des1, loc1 = sift(image1)
    im2, des2, loc2 = sift(image2)

    distRatio = 0.6

    n1 = des1.shape[0]
    n2 = des2.shape[0]
    matches = np.zeros((n1,), dtype=np.int64)  # 0 means no match, else 1-based index into image2 keypoints.

    if n1 > 0 and n2 >= 2:
        des2t = des2.T
        for i in range(n1):
            dotprods = des1[i, :] @ des2t
            dotprods = np.clip(dotprods, -1.0, 1.0)
            vals = np.arccos(dotprods)
            indx = np.argsort(vals)
            vals_sorted = vals[indx]

            if vals_sorted[0] < distRatio * vals_sorted[1]:
                matches[i] = int(indx[0]) + 1

    im3 = _to_float_gray(appendimages(im1, im2))
    overlay = np.repeat(im3[:, :, None], 3, axis=2)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float64)

    cols1 = im1.shape[1]
    for i in range(n1):
        if matches[i] > 0:
            j = matches[i] - 1
            _draw_line_rgb(
                overlay,
                float(loc1[i, 0]),
                float(loc1[i, 1]),
                float(loc2[j, 0]),
                float(loc2[j, 1] + cols1),
                yellow,
            )

    num = int(np.sum(matches > 0))
    print(f"Found {num} matches.")
    return num, overlay


__all__ = ["match"]
