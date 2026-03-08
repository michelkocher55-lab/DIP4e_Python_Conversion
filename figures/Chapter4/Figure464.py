import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.intScaling4e import intScaling4e
from libDIPUM.cnotch import cnotch
from libDIPUM.dftfilt import dftfilt
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('car-moire-pattern.tif')
f = img_as_float(imread(img_path))
M, N = f.shape

# DFT
F = np.fft.fft2(f)
S = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(F))))

# Notch locations (from impixelinfo)
C = np.array([[44, 54], [85, 56], [40, 112], [82, 112]])

# Notch filter (uncentered)
H = cnotch('butterworth', 'reject', M, N, C, 9, 4)

# Filtering
P = intScaling4e(np.fft.fftshift(H) * img_as_float(S))
g = np.real(dftfilt(f, H))
g = img_as_float(g)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(f, cmap='gray')
axes[0, 0].axis('off')

axes[0, 1].imshow(S, cmap='gray')
axes[0, 1].axis('off')

axes[1, 0].imshow(P, cmap='gray')
axes[1, 0].axis('off')

axes[1, 1].imshow(g, cmap='gray')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure464.png')
plt.show()
