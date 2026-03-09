from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import mean
from skimage import img_as_float
from skimage.io import imread
from skimage.segmentation import find_boundaries, slic

from libDIPUM.kmeans import kmeans
from General.superpixels import superpixels
from libDIPUM.data_path import dip_data


def mat2gray(img: Any):
    """mat2gray."""
    min_v = img.min()
    max_v = img.max()
    if max_v - min_v < 1e-12:
        return np.zeros_like(img, dtype=np.float64)
    return (img - min_v) / (max_v - min_v)


def slic_compat(image: Any, n_segments: Any):
    """slic_compat."""
    try:
        return slic(
            image,
            n_segments=n_segments,
            compactness=10,
            start_label=1,
            channel_axis=-1 if image.ndim == 3 else None,
        )
    except TypeError:
        return slic(
            image,
            n_segments=n_segments,
            compactness=10,
            start_label=1,
            multichannel=(image.ndim == 3),
        )


# Data
image_path = dip_data("book-cover.tif")
f_raw = imread(image_path)
f = img_as_float(f_raw)

# Super pixels
print("Computing superpixels (N=95000)...")
L, NL = superpixels(f, 95000)
L = np.asarray(L, dtype=np.int64)

# Validate segmentation quality for display and fallback if boundaries are excessive.
BW = find_boundaries(L, mode="inner")
if BW.mean() > 0.95:
    print(
        "Fallback: boundary mask is too dense; using SLIC superpixels for stable regions."
    )
    L = slic_compat(f, 95000)
    BW = find_boundaries(L, mode="inner")

print(f"Boundary pixels: {int(BW.sum())} / {BW.size} ({100.0 * BW.mean():.3f}%)")

labels = np.unique(L)
NL = int(labels.max())

# Replace each superpixel by its average value.
print("Computing means...")
if f.ndim == 3:
    fSP = np.zeros_like(f, dtype=np.float64)
    for c in range(f.shape[2]):
        means = mean(f[:, :, c], labels=L, index=labels)
        lut = np.zeros(NL + 1, dtype=np.float64)
        lut[labels] = means
        fSP[:, :, c] = lut[L]
else:
    means = mean(f, labels=L, index=labels)
    lut = np.zeros(NL + 1, dtype=np.float64)
    lut[labels] = means
    fSP = lut[L]

fSP = mat2gray(fSP)

# Segment fSP using k-means.
print("Running K-means on superpixel image (k=3)...")
km_input = fSP.mean(axis=2).ravel() if fSP.ndim == 3 else fSP.ravel()
idx_sp, _ = kmeans(km_input, 3)
fSPseg = idx_sp.reshape(fSP.shape[:2]).astype(np.float64)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(f, cmap="gray" if f.ndim == 2 else None)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(fSP, cmap="gray" if fSP.ndim == 2 else None)
axes[1].set_title("Superpixels Mean")
axes[1].axis("off")

axes[2].imshow(fSPseg, cmap="gray")
axes[2].set_title("Segmented Superpixels")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure1054.png")
print("Saved Figure1054.png")
plt.show()
