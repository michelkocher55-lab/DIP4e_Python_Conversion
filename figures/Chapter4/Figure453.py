import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.hpfilter import hpfilter
from libDIP.dftFiltering4e import dftFiltering4e
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('characterTestPattern688.tif')
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Ideal High Pass
H = np.fft.ifftshift(hpfilter('ideal', P, Q, 60))
gI60 = dftFiltering4e(f, H)
H = np.fft.ifftshift(hpfilter('ideal', P, Q, 160))
gI160 = dftFiltering4e(f, H)
gI160e = intScaling4e(gI160)

# Gaussian High Pass
H = np.fft.ifftshift(hpfilter('gaussian', P, Q, 60))
gG60 = dftFiltering4e(f, H)
H = np.fft.ifftshift(hpfilter('gaussian', P, Q, 160))
gG160 = dftFiltering4e(f, H)
gG160e = intScaling4e(gG160)

# Butterworth High Pass
H = np.fft.ifftshift(hpfilter('butterworth', P, Q, 60, 2))
gB60 = dftFiltering4e(f, H)
H = np.fft.ifftshift(hpfilter('butterworth', P, Q, 160, 2))
gB160 = dftFiltering4e(f, H)
gB160e = intScaling4e(gB160)

# Display figure 1
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
axes1[0, 0].imshow(gI60, cmap='gray', vmin=0, vmax=1)
axes1[0, 0].axis('off')

axes1[0, 1].imshow(gG60, cmap='gray', vmin=0, vmax=1)
axes1[0, 1].axis('off')

axes1[0, 2].imshow(gB60, cmap='gray', vmin=0, vmax=1)
axes1[0, 2].axis('off')

axes1[1, 0].imshow(gI160, cmap='gray', vmin=0, vmax=1)
axes1[1, 0].axis('off')

axes1[1, 1].imshow(gG160, cmap='gray', vmin=0, vmax=1)
axes1[1, 1].axis('off')

axes1[1, 2].imshow(gB160, cmap='gray', vmin=0, vmax=1)
axes1[1, 2].axis('off')

plt.tight_layout()
plt.savefig('Figure453.png')

# Display figure 2
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
axes2[0].imshow(gI160e, cmap='gray')
axes2[0].axis('off')

axes2[1].imshow(gG160e, cmap='gray')
axes2[1].axis('off')

axes2[2].imshow(gB160e, cmap='gray')
axes2[2].axis('off')

plt.tight_layout()
plt.savefig('Figure454.png')
plt.show()
