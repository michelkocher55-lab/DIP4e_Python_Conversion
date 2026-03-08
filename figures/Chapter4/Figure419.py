
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import resize
from scipy.ndimage import correlate
from libDIPUM.data_path import dip_data

print("Running Figure419 (Aliasing and Lowpass Filtering)...")

# Parameters
HSize = 5

# Data Loading
img_path = dip_data('barbara.tif')

f_orig = imread(img_path)
if f_orig.ndim == 3: f_orig = f_orig[:,:,0]
f = img_as_float(f_orig)

# Decimation
# fd = f (1:3:end, 1:3:end);
fd = f[::3, ::3]

# Interpolation nearest neighbor
# fdi = imresize(fd, 3, 'nearest');
# skimage resize takes output_shape.
# Output shape should match f if divisible by 3, or be 3x dims of fd.
# fd dimensions are ceil(M/3), ceil(N/3) roughly or floor depending on slicing.
# MATLAB slicing 1:3:end includes the first element.
# Python [::3] includes 0, 3, 6...
# Resulting shape:
target_shape = (fd.shape[0] * 3, fd.shape[1] * 3)
# Note: original f might handle boundaries differently.
# If f is 512x512, f[::3] len is 171. 171*3 = 513.
# So fdi might be slightly larger or smaller than f.
# I'll resize to f.shape if close, or just 3*fd.shape.
# MATLAB imresize(fd, 3) scales by factor 3.

fdi = resize(fd, target_shape, order=0, mode='edge', anti_aliasing=False)

# Low pass filtering
# h = fspecial ('average', HSize);
h = np.ones((HSize, HSize)) / (HSize * HSize)

# flp = imfilter (f, h, 'symmetric', 'same');
# 'symmetric' -> 'reflect'
flp = correlate(f, h, mode='reflect')

# Decimation of the low pass filtered image
# flpd = flp (1:3:end, 1:3:end);
flpd = flp[::3, ::3]

# Interpolation nearest neighbor
# flpdi = imresize(flpd, 3, 'nearest');
flpdi = resize(flpd, target_shape, order=0, mode='edge', anti_aliasing=False)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(fdi, cmap='gray')
axes[0, 1].set_title('Decimated, Interpolated NN')
axes[0, 1].axis('off')

axes[1, 0].imshow(flp, cmap='gray')
axes[1, 0].set_title('Low Pass Filtered')
axes[1, 0].axis('off')

axes[1, 1].imshow(flpdi, cmap='gray')
axes[1, 1].set_title('Smoothed, Decimated, Interpolated NN')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure419.png')
print("Saved Figure419.png")
plt.show()
