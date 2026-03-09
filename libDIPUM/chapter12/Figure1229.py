from typing import Any
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

print("Running Figure1229 (Histogram-based texture statistics)...")

# Data
files = [
    "OpticalMicroscope-superconductor-smooth-texture.tif",
    "OpticalMicroscope-cholesterol-rough-texture.tif",
    "OpticalMicroscope-microporcessor-regular-texture.tif",
]
names = ["Smooth", "Coarse", "Regular"]


def _to_gray_u8(img: Any):
    """_to_gray_u8."""
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        # preserve integer grayscale behavior from MATLAB hist 0..255
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _imcrop_matlab(img: Any, rect: Any):
    """_imcrop_matlab."""
    # MATLAB rect = [x, y, w, h], output includes both ends => (h+1, w+1)
    x, y, w, h = rect
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))
    return img[y : y + h + 1, x : x + w + 1]


f = []
for fn in files:
    img = imread(dip_data(fn))
    f.append(_to_gray_u8(img))

fc = [
    _imcrop_matlab(f[0], [80, 230, 68, 73]),
    _imcrop_matlab(f[1], [35, 162, 75, 80]),
    _imcrop_matlab(f[2], [16, 13, 64, 80]),
]

# Histogram based statistical analysis
z = np.arange(256, dtype=float)
m = np.zeros(len(fc), dtype=float)
mu2 = np.zeros(len(fc), dtype=float)
mu3 = np.zeros(len(fc), dtype=float)
NormVar = np.zeros(len(fc), dtype=float)
Uniformity = np.zeros(len(fc), dtype=float)
Entropy = np.zeros(len(fc), dtype=float)
p_list = []

for i, img in enumerate(fc):
    hist = np.bincount(img.ravel(), minlength=256).astype(float)
    p = hist / np.sum(hist)
    p_list.append(p)

    m[i] = np.sum(z * p)
    mu2[i] = np.sum(((z - m[i]) ** 2) * p)

    if mu2[i] > 0:
        mu3[i] = np.sum(((z - m[i]) ** 3) * p) / (mu2[i] ** 1.5)
    else:
        mu3[i] = 0.0

    NormVar[i] = mu2[i] / (255.0**2)
    Uniformity[i] = np.sum(p**2)
    nz = p > 0
    Entropy[i] = -np.sum(p[nz] * np.log2(p[nz]))

R = 1.0 - 1.0 / (1.0 + NormVar)

# Display
fig, ax = plt.subplots(3, 3, figsize=(12, 10))
ax = ax.ravel()

for i in range(len(f)):
    ax[i].imshow(f[i], cmap="gray")
    ax[i].set_title(names[i])
    ax[i].axis("off")

    ax[i + 3].imshow(fc[i], cmap="gray")
    ax[i + 3].set_title(f"μ={m[i]:.3g}, σ={np.sqrt(mu2[i]):.3g}, R = {R[i]:.3g}")
    ax[i + 3].axis("off")

    ax[i + 6].plot(z, p_list[i])
    ax[i + 6].set_xlim(0, 255)
    ax[i + 6].set_aspect("auto")
    ax[i + 6].set_xlabel("z")
    ax[i + 6].set_ylabel("p(z)")
    ax[i + 6].set_title(f"{mu3[i]:.3g}, {Uniformity[i]:.3g}, {Entropy[i]:.3g}")

fig.tight_layout()
fig.savefig("Figure1229.png")

print("Saved Figure1229.png")
plt.show()
