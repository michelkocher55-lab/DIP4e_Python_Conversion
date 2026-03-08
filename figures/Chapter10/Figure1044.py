
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import img_as_float
from libDIPUM.ishade import ishade
from libDIPUM.movingthresh import movingthresh
from libDIPUM.data_path import dip_data

# Data
image_path = dip_data('Fig1049(original_cursive_text_WITHOUT_SHADING).tif')
f_raw = imread(image_path)
if f_raw.ndim == 3:
    f_raw = f_raw[:, :, 0]

# Convert to float [0, 1]
f = img_as_float(f_raw)
M, N = f.shape

# Create shading pattern
shade = ishade(M, N, 0.1, 1.0, 'spot', min(M, N) / 2)

# Shade f
fs = f * shade

# Segment using Otsu's method
T = threshold_otsu(fs)
gotsu = fs > T

# Segment using moving average
n = 20
gmoving = movingthresh(fs, n, 0.5)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(fs, cmap='gray')
axes[0, 1].set_title('Shaded Image')
axes[0, 1].axis('off')

axes[1, 0].imshow(gotsu, cmap='gray')
axes[1, 0].set_title("Otsu's Method")
axes[1, 0].axis('off')

axes[1, 1].imshow(gmoving, cmap='gray')
axes[1, 1].set_title('Moving Average Threshold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure1044.png')
print("Saved Figure1044.png")
plt.show()