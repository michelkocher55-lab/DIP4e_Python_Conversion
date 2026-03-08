
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import img_as_float
from libDIPUM.imnoise3 import imnoise3
from libDIPUM.movingthresh import movingthresh
from libDIPUM.data_path import dip_data

# Data
image_path = dip_data('Fig1049(original_cursive_text_WITHOUT_SHADING).tif')
f_raw = imread(image_path)
if f_raw.ndim == 3:
    f_raw = f_raw[:, :, 0]

f = img_as_float(f_raw)
M, N = f.shape

# Create shading pattern
K = round(min(M, N) / 100.0)
C = np.array([[0, K]])
r, _, _ = imnoise3(M, N, C)

r = r - r.min()
r = r / (r.max() + 1e-10)
r = r + 0.25
r = r / (r.max() + 1e-10)

# Shade image
fs2 = f * r

# Segment using Otsu
T2 = threshold_otsu(fs2)
gotsu2 = fs2 > T2


# Segment using moving average
n = 20
gmoving2 = movingthresh(fs2, n, 0.5)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(fs2, cmap='gray')
axes[0, 1].set_title('Shaded Image (Sinusoidal)')
axes[0, 1].axis('off')

axes[1, 0].imshow(gotsu2, cmap='gray')
axes[1, 0].set_title("Otsu's Method")
axes[1, 0].axis('off')

axes[1, 1].imshow(gmoving2, cmap='gray')
axes[1, 1].set_title('Moving Average Threshold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure1045.png')
print("Saved Figure1045.png")
plt.show()