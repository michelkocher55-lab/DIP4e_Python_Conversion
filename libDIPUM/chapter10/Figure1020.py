import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve, uniform_filter
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("building-cropped-834by1114.tif")
f = img_as_float(imread(img_path))

# Kernels
Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

# Filtering (Smoothed)
fs = uniform_filter(f, size=5, mode="nearest")

# Gradient on Original
Gx = np.abs(convolve(f, Sx, mode="nearest"))
Gy = np.abs(convolve(f, Sy, mode="nearest"))
g = Gx + Gy

# Gradient on Smoothed
Gxs = np.abs(convolve(fs, Sx, mode="nearest"))
Gys = np.abs(convolve(fs, Sy, mode="nearest"))
gs = Gxs + Gys

# Thresholding
gt = g >= (0.33 * g.max())
gst = gs >= (0.33 * gs.max())

# Display
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.flatten()

axes[0].imshow(gt, cmap="gray")
axes[0].set_title("Thresholded Gradient (Original)")
axes[0].axis("off")

axes[1].imshow(gst, cmap="gray")
axes[1].set_title("Thresholded Gradient (Smoothed)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure1020.png")
print("Saved Figure1020.png")
plt.show()
