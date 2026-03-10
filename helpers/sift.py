"""SIFT keypoint extraction with MATLAB-like signature.

Equivalent interface to:
    [image, descriptors, locs] = sift(imageFile)

Returns
-------
image : ndarray, float64
    Grayscale image as float64.
descriptors : ndarray, shape (K, 128)
    SIFT descriptors, L2-normalized row-wise.
locs : ndarray, shape (K, 4)
    Keypoint attributes as (row, column, scale, orientation).
    Orientation is in radians in [-pi, pi].
"""

from __future__ import annotations
from typing import Any

import numpy as np


def _im2double(arr: np.ndarray) -> np.ndarray:
    """_im2double."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64)
    if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_:
        return a.astype(np.float64)
    return a.astype(np.float64)


def _to_float01(arr: np.ndarray) -> np.ndarray:
    """_to_float01."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        denom = float(info.max - info.min) if info.max != info.min else 1.0
        return (a.astype(np.float64) - float(info.min)) / denom
    if a.dtype == np.bool_:
        return a.astype(np.float64)

    out = a.astype(np.float64)
    if out.size == 0:
        return out
    mn = float(np.min(out))
    mx = float(np.max(out))
    if mn < 0.0:
        out = out - mn
        mx = float(np.max(out))
    if mx > 1.0:
        out = out / mx
    return np.clip(out, 0.0, 1.0)


def _to_u8_for_sift(arr: np.ndarray) -> np.ndarray:
    """_to_u8_for_sift."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        denom = float(info.max - info.min) if info.max != info.min else 1.0
        scaled = (a.astype(np.float64) - float(info.min)) / denom
        return np.uint8(np.clip(scaled, 0.0, 1.0) * 255.0)
    return np.uint8(_to_float01(a) * 255.0)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """_to_gray."""
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[..., 0]
        # RGB luminance conversion.
        return 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    raise ValueError("Input image must be 2-D grayscale or 3-D color.")


def _normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    """_normalize_descriptors."""
    if descriptors.size == 0:
        return descriptors.reshape(0, 128)
    d = descriptors.astype(np.float64, copy=False)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return d / n


def _locs_from_cv_keypoints(keypoints: Any) -> np.ndarray:
    """_locs_from_cv_keypoints."""
    if not keypoints:
        return np.zeros((0, 4), dtype=np.float64)

    k = len(keypoints)
    locs = np.zeros((k, 4), dtype=np.float64)

    for i, kp in enumerate(keypoints):
        # MATLAB format expects (row, col, scale, orientation).
        row = kp.pt[1]
        col = kp.pt[0]
        # OpenCV stores diameter in kp.size; sigma is approx size/2.
        scale = kp.size / 2.0
        angle_deg = ((kp.angle + 180.0) % 360.0) - 180.0
        ori = np.deg2rad(angle_deg)
        locs[i, :] = [row, col, scale, ori]

    return locs


def sift(imageFile: str):
    """Read an image and compute SIFT descriptors and keypoint locations.

    Parameters
    ----------
    imageFile : str
        Path to image file.

    Returns
    -------
    image : ndarray
        Grayscale float64 image in [0,1].
    descriptors : ndarray
        Kx128 descriptor matrix.
    locs : ndarray
        Kx4 matrix with rows (row, col, scale, orientation[rad]).
    """
    try:
        import imageio.v2 as iio
    except Exception as exc:
        raise ImportError("imageio is required to read images.") from exc

    image_raw = iio.imread(imageFile)
    gray_raw = _to_gray(image_raw)
    image = _im2double(gray_raw)

    # Backend 1: OpenCV SIFT (preferred)
    try:
        import cv2

        image_u8 = _to_u8_for_sift(gray_raw)
        detector = cv2.SIFT_create()
        keypoints, descriptors = detector.detectAndCompute(image_u8, None)

        if descriptors is None:
            return (
                image,
                np.zeros((0, 128), dtype=np.float64),
                np.zeros((0, 4), dtype=np.float64),
            )

        locs = _locs_from_cv_keypoints(keypoints)
        descriptors = _normalize_descriptors(descriptors)
        return image, descriptors, locs

    except Exception:
        pass

    # Backend 2: scikit-image SIFT fallback
    try:
        from skimage.feature import SIFT

        detector = SIFT()
        detector.detect_and_extract(_to_float01(gray_raw))

        keypoints = detector.keypoints  # rows, cols
        descriptors = detector.descriptors

        if keypoints is None or len(keypoints) == 0:
            return (
                image,
                np.zeros((0, 128), dtype=np.float64),
                np.zeros((0, 4), dtype=np.float64),
            )

        # scikit-image exposes scales and orientations with same ordering.
        scales = np.asarray(detector.scales, dtype=np.float64)
        oris = np.asarray(detector.orientations, dtype=np.float64)
        # Ensure orientation range [-pi, pi].
        oris = ((oris + np.pi) % (2.0 * np.pi)) - np.pi

        locs = np.column_stack(
            [
                keypoints[:, 0].astype(np.float64),
                keypoints[:, 1].astype(np.float64),
                scales,
                oris,
            ]
        )

        descriptors = _normalize_descriptors(descriptors)
        return image, descriptors, locs

    except Exception as exc:
        raise ImportError(
            "No SIFT backend available. Install opencv-contrib-python "
            "or a recent scikit-image with skimage.feature.SIFT."
        ) from exc


__all__ = ["sift"]
