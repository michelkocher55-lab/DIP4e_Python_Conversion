from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from skimage.segmentation import find_boundaries
from scipy.ndimage import mean
from General.superpixels import superpixels
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

# Superpixels sizes
NSUP = [1000, 500, 250]

fSPStore = []
BWStore = []

for i, n_seg in enumerate(NSUP):
    print(f"Processing superpixels N={n_seg}...")
    L, NL = superpixels(f, n_seg)

    # Mean image
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
    fSPStore.append(fSP)

    # Boundaries
    BW = find_boundaries(L, mode="thick")
    BWStore.append(BW)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(3):
    # Top row: Overlays
    fSP = fSPStore[i]
    BW = BWStore[i]

    if fSP.ndim == 2:
        f_overlay = fSP.copy()
        f_overlay[BW] = 1.0  # White
        cmap = "gray"
    else:
        f_overlay = fSP.copy()
        for c in range(3):
            f_overlay[:, :, c][BW] = 1.0
        cmap = None

    axes[0, i].imshow(f_overlay, cmap=cmap)
    axes[0, i].set_title(f"N={NSUP[i]} (Overlay)")
    axes[0, i].axis("off")

    # Bottom row: Mean Images
    axes[1, i].imshow(fSP, cmap=cmap)
    axes[1, i].set_title(f"N={NSUP[i]} (Mean)")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("Figure1052.png")
print("Saved Figure1052.png")
plt.show()
