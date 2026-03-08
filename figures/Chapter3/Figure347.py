import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import correlate1d
from libDIPUM.data_path import dip_data

# Image loading
img_path = dip_data('hickson-compact-group.tif')
f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]

f = img_as_float(f)

# Kernel (separable Gaussian, spatial domain)
SIG = 25.0
ksize = 151
radius = (ksize - 1) // 2
x = np.arange(-radius, radius + 1, dtype=np.float64)
w1d = np.exp(-(x**2) / (2.0 * SIG * SIG))
w1d /= np.sum(w1d)

# Filtering (equivalent to 2D Gaussian correlation, but much faster)
# MATLAB default imfilter boundary is zero padding.
g = correlate1d(f, w1d, axis=1, mode='constant', cval=0.0)
g = correlate1d(g, w1d, axis=0, mode='constant', cval=0.0)

# Thresholding
gT = g > 0.4

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(g, cmap='gray')
axes[1].set_title('Smoothed (Lowpass)')
axes[1].axis('off')

axes[2].imshow(gT, cmap='gray')
axes[2].set_title('Thresholded > 0.4')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure347.png')
print('Saved Figure347.png')
plt.show()
