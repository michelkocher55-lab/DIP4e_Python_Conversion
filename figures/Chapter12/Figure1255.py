"""Figure 12.55 - MSER of half-size building image."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as iio

from General.detectMSERFeatures import detectMSERFeatures
from libDIPUM.data_path import dip_data


def _to_gray(a: np.ndarray) -> np.ndarray:
    """_to_gray."""
    arr = np.asarray(a)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[..., 0]
        return (
            0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        ).astype(arr.dtype)
    raise ValueError("Input image must be 2-D grayscale or 3-D color")


def _to_uint8_graylevels(a: np.ndarray) -> np.ndarray:
    """_to_uint8_graylevels."""
    arr = np.asarray(a)
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float64)
    if out.size == 0:
        return np.zeros_like(out, dtype=np.uint8)
    if np.max(out) <= 1.0:
        out = out * 255.0
    return np.uint8(np.clip(np.round(out), 0.0, 255.0))


def _linear_idx_to_mask(
    pixel_idx_1based: np.ndarray, shape_hw: tuple[int, int]
) -> np.ndarray:
    """Convert MATLAB 1-based column-major linear indices to uint8 mask."""
    H, W = shape_hw
    out = np.zeros((H, W), dtype=np.uint8)
    idx0 = np.asarray(pixel_idx_1based, dtype=np.int64).ravel() - 1
    idx0 = idx0[(idx0 >= 0) & (idx0 < H * W)]
    if idx0.size == 0:
        return out
    rr, cc = np.unravel_index(idx0, (H, W), order="F")
    out[rr, cc] = 255
    return out


print("Running Figure1255 (MSER of half-size building image)...")

# Data
img_path = dip_data("building-600by600.tif")
I = _to_gray(iio.imread(img_path))

# Quarter size in pixels (half width and height): I(1:2:end, 1:2:end)
I = I[::2, ::2]

# Kernel
w = np.ones((3, 3), dtype=np.float64)
w = w / np.sum(w)

# Filtering
Is = ndimage.convolve(I, w, mode="nearest")
Is_u8 = _to_uint8_graylevels(Is)

# MSER
R, RCC = detectMSERFeatures(
    Is_u8,
    "ThresholdDelta",
    0.7,
    "RegionAreaRange",
    [2500, 7500],
)

CC = RCC["PixelIdxList"]
NCC = len(CC)
Iunion = np.zeros_like(Is_u8, dtype=np.uint8)
for K in range(NCC):
    Idisp = _linear_idx_to_mask(CC[K], Is_u8.shape)
    Iunion = np.bitwise_or(Iunion, Idisp)

# Display
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(I, cmap="gray")
axs[0].set_title("I half size")
axs[0].axis("off")

axs[1].imshow(Is_u8, cmap="gray")
axs[1].set_title("Is")
axs[1].axis("off")

axs[2].imshow(Iunion, cmap="gray")
axs[2].set_title("MSER")
axs[2].axis("off")

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1255.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Detected regions: {R.Count}")
print(f"Saved {out_path}")
plt.show()
