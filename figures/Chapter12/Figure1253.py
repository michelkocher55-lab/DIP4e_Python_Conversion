"""Figure 12.53 - MSER of building."""

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


print("Running Figure1253 (MSER of building)...")

# Data
img_path = dip_data("building-600by600.tif")
I = _to_gray(iio.imread(img_path))

# Kernel
w = np.ones((5, 5), dtype=np.float64)
w = w / np.sum(w)

# Filtering (replicate boundary)
Is = ndimage.convolve(I, w, mode="nearest")

# MSER
R, RCC = detectMSERFeatures(
    Is,
    "ThresholdDelta",
    0.4,
    "RegionAreaRange",
    [10000, 30000],
)

# Display all images.
CC = RCC["PixelIdxList"]
NCC = len(CC)
Iunion = np.zeros_like(I, dtype=np.uint8)

fig = plt.figure(figsize=(12, 8))
for K in range(NCC):
    Idisp = _linear_idx_to_mask(CC[K], I.shape)
    Iunion = np.bitwise_or(Iunion, Idisp)

    subplot_idx = K + 1  # MATLAB: subplot(2,3,K)
    if subplot_idx > 6:
        break
    ax = fig.add_subplot(2, 3, subplot_idx)
    ax.imshow(Idisp, cmap="gray")
    ax.set_title(f"Region {K + 1}")
    ax.axis("off")

ax = fig.add_subplot(2, 3, 6)  # MATLAB: subplot(2,3,6)
ax.imshow(Iunion, cmap="gray")
ax.set_title("Union")
ax.axis("off")

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1253.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Detected regions: {R.Count}")
print(f"Saved {out_path}")
plt.show()
