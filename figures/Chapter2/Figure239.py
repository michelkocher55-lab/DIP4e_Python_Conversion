
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import uniform_filter
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('angiogram-aortic-kidney.tif')

# Load image
f = imread(img_path)

# Averaging
# w = fspecial('average', 41);
# g = imfilter(f, w, 'symmetric');
# 'average' filter is a uniform filter.
# 'symmetric' padding in MATLAB is equivalent to 'reflect' in scipy.ndimage.

# Note: uniform_filter handles different types. If f is integer, it mimics integer arithmetic unless computed as float.
# MATLAB: imfilter with uint8 input and double kernel 'average' -> Output is double? Or same class?
# Actually imfilter returns same class as input usually, but if result exceeds range it might truncate?
# fspecial returns doubles.
# If input is uint8, imfilter converts to double for calculation, then casts back to uint8 (rounding).
# Let's use float for calculation.

f_float = f.astype(float)
g_float = uniform_filter(f_float, size=41, mode='reflect')

# Convert back to uint8 for display if original was uint8, or keep as is.
# MATLAB imshow handles class.
# We will cast to uint8 for consistency with MATLAB result which likely returns uint8 if input was uint8.
g = np.clip(g_float, 0, 255).astype(f.dtype)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()

axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(g, cmap='gray')
axes[1].set_title('Smoothed Image (41x41 Average)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Figure239.png')
print("Saved Figure239.png")

plt.show()