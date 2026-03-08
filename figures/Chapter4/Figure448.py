import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.lpfilter import lpfilter
from libDIP.dftFiltering4e import dftFiltering4e
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('text_gaps_of_1_and_2_pixels.tif')
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Filter design in the frequency domain
H = np.fft.fftshift(lpfilter('gaussian', P, Q, 120))

# Filtering in the frequency domain
g = dftFiltering4e(f, H)
gs = intScaling4e(g)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f, cmap='gray')
axes[0].axis('off')

axes[1].imshow(gs, cmap='gray')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Figure448.png')
plt.show()
