from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from skimage.segmentation import find_boundaries
from scipy.ndimage import mean
from helpers.superpixels import superpixels
from helpers.kmeans import kmeans
from libDIPUM.data_path import dip_data


def mat2gray(img: Any):
    """mat2gray."""
    min_v = img.min()
    max_v = img.max()
    if max_v - min_v < 1e-10:
        return np.zeros_like(img)
    return (img - min_v) / (max_v - min_v)


# Data
image_path = dip_data("iceberg.tif")
f_raw = imread(image_path)
f = img_as_float(f_raw)

# K-means
print("Running initial K-means (k=3)...")
idx_raw, _ = kmeans(f.flatten(), 3)

fseg = idx_raw.reshape(f.shape).astype(float)
fseg = mat2gray(fseg)

# Superpixels
print("Computing superpixels (N=100)...")
L, NL = superpixels(f, 100)

# Replace each superpixel by its average value
print("Computing means...")
if f.ndim == 3:
    fSP = np.zeros_like(f)
    for c in range(f.shape[2]):
        means = mean(f[:, :, c], labels=L, index=np.arange(1, NL + 1))
        mapping = np.zeros(NL + 1)
        mapping[1:] = means
        fSP[:, :, c] = mapping[L]
else:
    means = mean(f, labels=L, index=np.arange(1, NL + 1))
    mapping = np.zeros(NL + 1)
    mapping[1:] = means
    fSP = mapping[L]

fSP = mat2gray(fSP)

# BW = boundarymask(L);
BW = find_boundaries(L, mode="thick")

print("Running K-means on superpixel image (k=3)...")
idx_sp, _ = kmeans(fSP.flatten(), 3)

fSPseg = idx_sp.reshape(fSP.shape).astype(float)
fSPseg = mat2gray(fSPseg)

# Display

# Prepare overlay
if fSP.ndim == 2:
    f_overlay = fSP.copy()
    f_overlay[BW] = 1.0  # White
    cmap = "gray"
else:
    f_overlay = fSP.copy()
    for c in range(3):
        f_overlay[:, :, c][BW] = 1.0
    cmap = None

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(f, cmap=cmap)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(fseg, cmap=cmap)
axes[0, 1].set_title("k-means")
axes[0, 1].axis("off")

axes[0, 2].axis("off")

axes[1, 0].imshow(f_overlay, cmap=cmap)
axes[1, 0].set_title("Superpixels Overlay")
axes[1, 0].axis("off")

axes[1, 1].imshow(fSP, cmap=cmap)
axes[1, 1].set_title("Superpixels Mean")
axes[1, 1].axis("off")

axes[1, 2].imshow(fSPseg, cmap=cmap)
axes[1, 2].set_title("Segmented Superpixels")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Figure1053.png")
print("Saved Figure1053.png")
plt.show()
