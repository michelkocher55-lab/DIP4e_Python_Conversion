import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve as ndi_convolve
from skimage.io import imread
from skimage.util import img_as_float
from General.fspecial import fspecial
from libDIPUM.data_path import dip_data

# Figure 10.22 (Marr-Hildreth)
def log_zero_cross_edges(f, sigma, threshold):
    """
    LoG edge detector with thresholded zero crossings.
    threshold=0 gives characteristic closed-loop edges.
    """
    hsize = int(np.ceil(6 * sigma))
    if hsize % 2 == 0:
        hsize += 1

    h = fspecial('log', hsize, sigma)
    log_img = ndi_convolve(f, h, mode='nearest')

    # Zero-crossings in 4-neighborhood with transition-strength threshold.
    out = np.zeros_like(log_img, dtype=bool)

    a = log_img[:-1, :]
    b = log_img[1:, :]
    zc = (a * b) < 0
    strong = np.abs(a - b) > threshold
    out[:-1, :] |= (zc & strong)

    a = log_img[:, :-1]
    b = log_img[:, 1:]
    zc = (a * b) < 0
    strong = np.abs(a - b) > threshold
    out[:, :-1] |= (zc & strong)

    return log_img, out

# Data
img_path = dip_data('building-cropped-834by1114.tif')
f = img_as_float(imread(img_path))

# Generate LoG filter for display (as in MATLAB script).
LoGfilter = fspecial('log', 25, 4)

# Filtering
LoGimage = ndi_convolve(f, LoGfilter, mode='nearest')
print(np.max(LoGimage))
print(np.min(LoGimage))

# Zero crossings with thresholds 0 and 0.0009
gTzero = log_zero_cross_edges(f, sigma=4, threshold=0.0)[1]
gT0009 = log_zero_cross_edges(f, sigma=4, threshold=0.0009)[1]

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

axes[0].imshow(f, cmap='gray')
axes[0].axis('off')

axes[1].imshow(LoGimage, cmap='gray')
axes[1].axis('off')

axes[2].imshow(gTzero, cmap='gray')
axes[2].axis('off')

axes[3].imshow(gT0009, cmap='gray')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('Figure1022.png', dpi=300, bbox_inches='tight')
plt.show()
