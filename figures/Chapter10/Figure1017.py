import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("building-cropped-834by1114.tif")
f = img_as_float(imread(img_path))

# Kernel
Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

# Gradient
Gx = convolve(f, Sx, mode="nearest")
Gy = convolve(f, Sy, mode="nearest")

# angle = atan2(Gy,Gx);
angle = np.arctan2(Gy, Gx)

# Display
# imshow(angle, []) scales min-max.

plt.figure(figsize=(8, 8))
plt.imshow(angle, cmap="gray")
plt.axis("off")
plt.title("Gradient Angle")

plt.savefig("Figure1017.png")
print("Saved Figure1017.png")
plt.show()
