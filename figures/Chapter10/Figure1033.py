
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float, random_noise
from libDIPUM.data_path import dip_data

print("Running Figure1033 (Histograms of noisy images)...")

# Data
img_path = dip_data('Fig1036(a)(original_septagon).tif')
f_orig = imread(img_path)

# Ensure standard grayscale
if f_orig.ndim == 3:
    f_orig = f_orig[:, :, 0]

f = img_as_float(f_orig)

# Add Gaussian noise
fn1 = random_noise(f, mode='gaussian', mean=0, var=0.002)
fn2 = random_noise(f, mode='gaussian', mean=0, var=0.038)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Images
axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(fn1, cmap='gray')
axes[1].set_title('Gaussian Noise (var=0.002)')
axes[1].axis('off')

axes[2].imshow(fn2, cmap='gray')
axes[2].set_title('Gaussian Noise (var=0.038)')
axes[2].axis('off')

# Histograms
# MATLAB: imhist(f) -> 256 bins for uint8 usually.
# For float images, it might use 256 bins mapped to [0,1].

axes[3].hist(f.ravel(), bins=256, range=(0, 1), color='black', alpha=0.7)
axes[3].set_title('Histogram (Original)')
axes[3].set_xlim([0, 1])

axes[4].hist(fn1.ravel(), bins=256, range=(0, 1), color='black', alpha=0.7)
axes[4].set_title('Histogram (Noise var=0.002)')
axes[4].set_xlim([0, 1])

axes[5].hist(fn2.ravel(), bins=256, range=(0, 1), color='black', alpha=0.7)
axes[5].set_title('Histogram (Noise var=0.038)')
axes[5].set_xlim([0, 1])

plt.tight_layout()
plt.savefig('Figure1033.png')
print("Saved Figure1033.png")
plt.show()