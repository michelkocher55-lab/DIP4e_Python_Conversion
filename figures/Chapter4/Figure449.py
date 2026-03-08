import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.lpfilter import lpfilter
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('woman512x512.tif')
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# MATLAB imcrop([256,217,107,73]) is [x, y, width, height] with 1-based
# Python slices: rows y:(y+height), cols x:(x+width)
x, y, w, h = 256, 217, 107, 73
fCrop = f[y:y + h, x:x + w]

# Filter design
H150 = np.fft.fftshift(lpfilter('gaussian', P, Q, 150))
H130 = np.fft.fftshift(lpfilter('gaussian', P, Q, 130))

# Filtering
g150 = dftFiltering4e(f, H150)
g150Crop = g150[y:y + h, x:x + w]

g130 = dftFiltering4e(f, H130)
g130Crop = g130[y:y + h, x:x + w]

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(f, cmap='gray')
axes[0, 0].axis('off')

axes[0, 1].imshow(g150, cmap='gray')
axes[0, 1].axis('off')

axes[0, 2].imshow(g130, cmap='gray')
axes[0, 2].axis('off')

axes[1, 0].imshow(fCrop, cmap='gray')
axes[1, 0].axis('off')

axes[1, 1].imshow(g150Crop, cmap='gray')
axes[1, 1].axis('off')

axes[1, 2].imshow(g130Crop, cmap='gray')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('Figure449.png')
plt.show()
