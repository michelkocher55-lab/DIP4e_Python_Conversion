"""Figure 12.58 - Difference of Gaussians."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from libDIPUM.data_path import dip_data


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """gaussian_kernel."""
    size = int(size)
    if size < 1:
        size = 1
    # Match fspecial-style centered kernel for odd/even sizes.
    r = (size - 1) / 2.0
    x = np.linspace(-r, r, size)
    y = np.linspace(-r, r, size)
    xx, yy = np.meshgrid(x, y)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    s = k.sum()
    if s != 0:
        k = k / s
    return k


def im2uint8_from_double(a: np.ndarray) -> np.ndarray:
    """im2uint8_from_double."""
    # MATLAB im2uint8 behavior for double input: clamp to [0,1], then scale.
    return np.uint8(np.clip(a, 0.0, 1.0) * 255.0)


def im2double(a: np.ndarray) -> np.ndarray:
    """im2double."""
    arr = np.asarray(a)
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float64)
        # Defensive normalization for float images encoded in [0,255].
        if out.size and out.max() > 1.0:
            out = out / 255.0
        return out
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.min < 0:
            return (arr.astype(np.float64) - info.min) / (info.max - info.min)
        return arr.astype(np.float64) / info.max
    if arr.dtype == np.bool_:
        return arr.astype(np.float64)
    return arr.astype(np.float64)


print("Running Figure1258...")

# Parameters
k = np.sqrt(2.0)
sdev = k / 2.0
T = 3

# Data
img_path = dip_data("building-600by600.tif")
f1 = plt.imread(img_path)
if f1.ndim == 3:
    f1 = f1[..., 0]
f1 = im2double(f1)

nr, nc = f1.shape
f2 = f1[::2, ::2]
f3 = f2[::2, ::2]

# Octave 1
sig = np.zeros(5, dtype=np.float64)
sig[0] = sdev
sig[1] = k * sig[0]
for i in range(2, 5):
    sig[i] = k * sig[i - 1]

oct1 = np.zeros((f1.shape[0], f1.shape[1], 5), dtype=np.float64)
for i in range(5):
    w = gaussian_kernel(int(np.ceil(6 * sig[i])), sig[i])
    oct1[:, :, i] = ndimage.convolve(f1, w, mode="nearest")

# Octave 2
sig[0] = 2 * sdev
sig[1] = k * sig[0]
for i in range(2, 5):
    sig[i] = k * sig[i - 1]

oct2 = np.zeros((f2.shape[0], f2.shape[1], 5), dtype=np.float64)
for i in range(5):
    w = gaussian_kernel(int(np.ceil(6 * sig[i])), sig[i])
    oct2[:, :, i] = ndimage.convolve(f2, w, mode="nearest")

# Octave 3
sig[0] = 4 * sdev
sig[1] = k * sig[0]
for i in range(2, 5):
    sig[i] = k * sig[i - 1]

oct3 = np.zeros((f3.shape[0], f3.shape[1], 5), dtype=np.float64)
for i in range(5):
    w = gaussian_kernel(int(np.ceil(3 * sig[i])), sig[i])
    oct3[:, :, i] = ndimage.convolve(f3, w, mode="nearest")

# Difference of Gaussians
DoG1 = oct1[:, :, 2] - oct1[:, :, 1]
DoG2 = oct2[:, :, 2] - oct2[:, :, 1]
DoG3 = oct3[:, :, 2] - oct3[:, :, 1]

DoG18 = im2uint8_from_double(DoG1)
DoG28 = im2uint8_from_double(DoG2)
DoG38 = im2uint8_from_double(DoG3)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Keep the same order as MATLAB code.
for ax, img in zip(axes, [DoG38 > T, DoG28 > T, DoG18 > T]):
    h, w = img.shape
    ax.imshow(
        img,
        cmap="gray",
        interpolation="nearest",
        origin="upper",
        extent=(1, w, h, 1),
    )
    ax.set_xlim(1, nc)
    ax.set_ylim(nr, 1)
    ax.axis("off")

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1258.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()
