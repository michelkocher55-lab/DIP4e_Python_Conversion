
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve, uniform_filter
from libDIP.edgeKernel4e import edgeKernel4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('building-cropped-834by1114.tif')
f = img_as_float(imread(img_path))

# Kirsch Kernels
wK45 = edgeKernel4e('kirsch', 'nw')
wKm45 = edgeKernel4e('kirsch', 'sw')

# Smoothing
fs = uniform_filter(f, size=5, mode='nearest')

# Filter
G45 = convolve(fs, wK45, mode='nearest')
Gm45 = convolve(fs, wKm45, mode='nearest')

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()

axes[0].imshow(G45, cmap='gray')
axes[0].set_title('NW Kirsch (45 deg)')
axes[0].axis('off')

axes[1].imshow(Gm45, cmap='gray')
axes[1].set_title('SW Kirsch (-45 deg)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Figure1019.png')
print("Saved Figure1019.png")
plt.show()