from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve as ndi_convolve
from skimage.io import imread
from skimage.util import img_as_float
from General.edge import edge
from General.fspecial import fspecial
from libDIPUM.data_path import dip_data

# Figure 10.25
# Canny edge detection of building


def log_zero_cross_edges(f: Any, sigma: Any, threshold: Any):
    """
    LoG edge detector with thresholded zero crossings.
    """
    hsize = int(np.ceil(6 * sigma))
    if hsize % 2 == 0:
        hsize += 1

    h = fspecial("log", hsize, sigma)
    log_img = ndi_convolve(f, h, mode="nearest")

    out = np.zeros_like(log_img, dtype=bool)

    a = log_img[:-1, :]
    b = log_img[1:, :]
    zc = (a * b) < 0
    strong = np.abs(a - b) > threshold
    out[:-1, :] |= zc & strong

    a = log_img[:, :-1]
    b = log_img[:, 1:]
    zc = (a * b) < 0
    strong = np.abs(a - b) > threshold
    out[:, :-1] |= zc & strong

    return log_img, out


# Data
img_path = dip_data("Fig1016(a)(building_original).tif")
f = img_as_float(imread(img_path))
if f.ndim == 3:
    f = f[..., 0]

# Kernels
w = fspecial("average", 5)
Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

# Filtering
fs = ndi_convolve(f, w, mode="nearest")

Gx = np.abs(ndi_convolve(f, Sx, mode="nearest"))
Gy = np.abs(ndi_convolve(f, Sy, mode="nearest"))
g = Gx + Gy

Gxs = np.abs(ndi_convolve(fs, Sx, mode="nearest"))
Gys = np.abs(ndi_convolve(fs, Sy, mode="nearest"))
gs = Gxs + Gys

# Thresholding
gt = g >= 0.33 * np.max(g)
gst = gs >= 0.33 * np.max(gs)

# LoG filter and filtering
LoGfilter = fspecial("log", 25, 4)
LoGimage = ndi_convolve(f, LoGfilter, mode="nearest")
print(np.max(LoGimage))
print(np.min(LoGimage))

# Marr-Hildreth and Canny
gT0009 = log_zero_cross_edges(f, sigma=4, threshold=0.0009)[1]
gcanny = edge(f, "canny", [0.04, 0.1], 4)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(gst, cmap="gray")
axes[1].set_title("Thresholded gradient of smoothed image")
axes[1].axis("off")

axes[2].imshow(gT0009, cmap="gray")
axes[2].set_title("Marr-Hildreth")
axes[2].axis("off")

axes[3].imshow(gcanny, cmap="gray")
axes[3].set_title("Canny")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("Figure1025.png", dpi=300, bbox_inches="tight")
plt.show()
