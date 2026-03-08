import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.hpfilter import hpfilter
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('thumb-print.tif')
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Filter design
H = np.fft.ifftshift(hpfilter('butterworth', P, Q, 50, 4))

# Filtering
g = dftFiltering4e(f, H)

# Thresholding
gp = g >= 0

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(f, cmap='gray')
axes[0].axis('off')

axes[1].imshow(g, cmap='gray')
axes[1].axis('off')

axes[2].imshow(gp, cmap='gray')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure455.png')
plt.show()
