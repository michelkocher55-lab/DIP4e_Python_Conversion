import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.exposure import equalize_hist
from libDIPUM.hpfilter import hpfilter
from libDIP.dftFiltering4e import dftFiltering4e
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('chestXray.tif')
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# High pass filter design
H = np.fft.ifftshift(hpfilter('gaussian', P, Q, 70))

# High frequency emphasis filter design
Hemp = 0.5 + 0.75 * H

# High pass filtering
ghp = dftFiltering4e(f, H)
ghps = intScaling4e(ghp)

# High frequency emphasis filtering
gemp = dftFiltering4e(f, Hemp)
gemps = intScaling4e(gemp)

# Histogram equalization (256 bins)
geq = equalize_hist(gemp, nbins=256)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(f, cmap='gray')
axes[0, 0].axis('off')

axes[0, 1].imshow(ghps, cmap='gray')
axes[0, 1].axis('off')

axes[1, 0].imshow(gemps, cmap='gray')
axes[1, 0].axis('off')

axes[1, 1].imshow(geq, cmap='gray')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure457.png')
plt.show()
