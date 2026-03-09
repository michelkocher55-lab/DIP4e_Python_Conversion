import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import correlate
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Data
f = imread(dip_data("blurry-moon.tif"))
if f.ndim == 3:
    f = f[:, :, 0]
f = img_as_float(f)

# Convolution kernels
w4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
w8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=float)

# Spatial filtering (MATLAB imfilter default: zero padding)
lap4 = correlate(f, w4, mode="constant", cval=0.0)
lap4s = intScaling4e(lap4)

lap8 = correlate(f, w8, mode="constant", cval=0.0)

# Enhanced images
g4 = f - lap4
g8 = f - lap8

# Display figure 1
fig1 = plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.imshow(f, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(np.abs(lap4), cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(g4, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(g8, cmap="gray")
plt.axis("off")

plt.tight_layout()
fig1.savefig("Figure352.png")
print("Saved Figure352.png")

# Display figure 2
fig2 = plt.figure(figsize=(6, 6))
plt.imshow(lap4s, cmap="gray")
plt.axis("off")
plt.tight_layout()
fig2.savefig("Figure353.png")
print("Saved Figure353.png")

plt.show()
