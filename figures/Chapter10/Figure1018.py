import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve, uniform_filter
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("building-cropped-834by1114.tif")
f = img_as_float(imread(img_path))

# Kernel
fs = uniform_filter(f, size=5, mode="nearest")

# Sobel Kernels
Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

# Gradient on Smoothed Image
Gxs = np.abs(convolve(fs, Sx, mode="nearest"))
Gys = np.abs(convolve(fs, Sy, mode="nearest"))

# gs = Gxs + Gys;
gs = Gxs + Gys

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(fs, cmap="gray")
axes[0].set_title("Smoothed Image (fs)")
axes[0].axis("off")

axes[1].imshow(Gxs, cmap="gray")
axes[1].set_title("Gx of Smoothed (Gxs)")
axes[1].axis("off")

axes[2].imshow(Gys, cmap="gray")
axes[2].set_title("Gy of Smoothed (Gys)")
axes[2].axis("off")

axes[3].imshow(gs, cmap="gray")
axes[3].set_title("Gradient of Smoothed (gs)")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("Figure1018.png")
print("Saved Figure1018.png")
plt.show()
