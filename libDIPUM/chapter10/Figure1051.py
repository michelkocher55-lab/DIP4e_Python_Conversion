from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from scipy.ndimage import mean
from helpers.superpixels import superpixels
from libDIPUM.data_path import dip_data


def mat2gray(img: Any):
    """mat2gray."""
    min_v = img.min()
    max_v = img.max()
    if max_v - min_v < 1e-10:
        return np.zeros_like(img)
    return (img - min_v) / (max_v - min_v)


# Data
image_path = dip_data("totem-poles.tif")
f_raw = imread(image_path)
f = img_as_float(f_raw)

# Get superpixel labels
print("Computing superpixels (N=40000)...")
L, NL = superpixels(f, 40000)

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

# Difference image
diff = mat2gray(f - fSP)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cmap = "gray" if f.ndim == 2 else None

axes[0].imshow(f, cmap=cmap)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(fSP, cmap=cmap)
axes[1].set_title("Superpixels Mean Image")
axes[1].axis("off")

axes[2].imshow(diff, cmap=cmap)
axes[2].set_title("Difference Image")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure1051.png")
print("Saved Figure1051.png")
plt.show()
