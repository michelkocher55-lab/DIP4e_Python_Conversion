from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.io import imread
from skimage.transform import resize
from libDIPUM.nCutSegmentation import nCutSegmentation
from libDIPUM.mat2gray import mat2gray
from libDIPUM.data_path import dip_data


def remap_two_regions(labels: Any, ref_image: Any):
    """Map 2 labels to {0,1} with bottom-center region forced to white."""
    u = np.unique(labels)
    if u.size != 2:
        raise RuntimeError(f"Expected 2 regions, got {u.size}")

    r0 = int(0.85 * labels.shape[0])
    c0 = int(0.50 * labels.shape[1])
    fg_label = labels[r0, c0]
    out = np.zeros_like(labels, dtype=np.float64)
    out[labels == fg_label] = 1.0
    return out


def remap_three_regions(labels: Any, ref_image: Any):
    """Map 3 labels to {0,0.5,1} ordered by mean brightness."""
    u = np.unique(labels)
    if u.size != 3:
        raise RuntimeError(f"Expected 3 regions, got {u.size}")

    means = np.array([ref_image[labels == lab].mean() for lab in u], dtype=np.float64)
    order = np.argsort(means)  # darkest -> brightest

    out = np.zeros_like(labels, dtype=np.float64)
    out[labels == u[order[0]]] = 0.0
    out[labels == u[order[1]]] = 0.5
    out[labels == u[order[2]]] = 1.0
    return out


def refine_two_region_from_lowres(S2: Any, I: Any):
    """Figure1058-like refinement: robust low-res cut + sharp full-res threshold."""
    # Orient labels with a spatial cue (iceberg near lower half).
    S2_bin = remap_two_regions(S2, I).astype(bool)
    # Transfer low-res class means to full-res smoothed image.
    I_low = resize(I, S2.shape, order=1, preserve_range=True, anti_aliasing=True)
    fg_mean = float(I_low[S2_bin].mean())
    bg_mean = float(I_low[~S2_bin].mean())
    t = 0.5 * (fg_mean + bg_mean)
    if fg_mean >= bg_mean:
        out = (I >= t).astype(np.float64)
    else:
        out = (I < t).astype(np.float64)
    return out


# Data
image_path = dip_data("iceberg.tif")
f_raw = imread(image_path)
if f_raw.ndim == 3:
    f = f_raw.mean(axis=2)
else:
    f = f_raw.astype(np.float64)
f = mat2gray(f)

# Smooth with 25x25 box kernel.
print("Applying smoothing (25x25)...")
I = uniform_filter(f, size=25, mode="nearest")

# Graph-cut segmentation with 2 and 3 regions.
print("Running graph cut (2 regions, sf=0.35)...")
S2 = nCutSegmentation(I, 2, sf=0.35, n_segments=1200)
S2v = refine_two_region_from_lowres(S2, I)

print("Running graph cut (3 regions, sf=0.35)...")
S3 = nCutSegmentation(I, 3, sf=0.35, n_segments=1200)
S3v = remap_three_regions(S3, I)

# Display
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1)
axes[0, 0].axis("off")

axes[0, 1].imshow(I, cmap="gray", vmin=0, vmax=1)
axes[0, 1].axis("off")

axes[1, 0].imshow(S2v, cmap="gray", vmin=0, vmax=1)
axes[1, 0].axis("off")

axes[1, 1].imshow(S3v, cmap="gray", vmin=0, vmax=1)
axes[1, 1].axis("off")

plt.tight_layout(pad=0.5)
plt.savefig("Figure1059.png", dpi=150)
print("Saved Figure1059.png")
plt.show()
