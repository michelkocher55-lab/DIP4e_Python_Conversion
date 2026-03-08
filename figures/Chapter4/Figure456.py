import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.dftuv import dftuv
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('blurry-moon.tif')
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Transfer function
U, V = dftuv(P, Q)
H = -4 * (np.pi ** 2) * (U ** 2 + V ** 2)
H = np.fft.fftshift(H)

# Filtering
glap = dftFiltering4e(f, H)

# Scale Laplacian response
glaps = glap / np.max(glap)

# Sharpened image
g = f - glaps

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f, cmap='gray')
axes[0].axis('off')

axes[1].imshow(g, cmap='gray')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Figure456.png')
plt.show()
