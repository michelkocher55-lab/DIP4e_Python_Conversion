
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('building-cropped-834by1114.tif')
f = img_as_float(imread(img_path))

# Kernel
Sx = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=float)

Sy = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=float)

# Filtering
Gx = np.abs(convolve(f, Sx, mode='nearest'))
Gy = np.abs(convolve(f, Sy, mode='nearest'))

# g = Gx + Gy;
g = Gx + Gy

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap='gray')
axes[0].set_title('f')
axes[0].axis('off')

axes[1].imshow(Gx, cmap='gray')
axes[1].set_title('Gx (Response to Sx)')
axes[1].axis('off')

axes[2].imshow(Gy, cmap='gray')
axes[2].set_title('Gy (Response to Sy)')
axes[2].axis('off')

axes[3].imshow(g, cmap='gray')
axes[3].set_title('Gradient Magnitude (Gx + Gy)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('Figure1016.png')
print(f"Saved Figure1016.png using {img_path}")
plt.show()