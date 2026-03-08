
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve, uniform_filter
from General.edge import edge
from General.fspecial import fspecial
from libDIPUM.data_path import dip_data
print("Running Figure1026 (Edge Detection Comparison)...")


def log_zero_cross_edges(f, sigma, threshold):
    """
    LoG edge detector with thresholded zero crossings.
    """
    hsize = int(np.ceil(6 * sigma))
    if hsize % 2 == 0:
        hsize += 1

    h = fspecial('log', hsize, sigma)
    log_img = convolve(f, h, mode='nearest')

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
img_path = dip_data('headCT.tif')
f = img_as_float(imread(img_path))

# 1. Smoothed Gradient
# w = fspecial('average', 5);
# fs = imfilter(f,w,'replicate');
fs = uniform_filter(f, size=5, mode='nearest')

# Sobel Masks
# Sx = [-1 -2 -1; 0 0 0; 1 2 1];
# Sy = [-1 0 1; -2 0 2; -1 0 1];
Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

# Gxs = abs(imfilter(fs,Sx,'conv','replicate'));
# Gys = abs(imfilter(fs,Sy,'conv','replicate'));
Gxs = np.abs(convolve(fs, Sx, mode='nearest'))
Gys = np.abs(convolve(fs, Sy, mode='nearest'))

# gs = Gxs + Gys; % Gradient image.
gs = Gxs + Gys

# gst=gs>=0.15*max(gs(:));
gst = gs >= (0.15 * gs.max())

# 2. Marr-Hildreth edge detection.
# gm = edge(f,'log',0.002,3);
# Note: 0.002 threshold. Sigma=3.
gm = log_zero_cross_edges(f, sigma=3, threshold=0.002)[1]

# 3. Canny edge detection
# gcan = edge(f,'canny',[0.05 0.15], 2);
gcan = edge(f, 'canny', [0.05, 0.15], 2)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original Head CT')
axes[0].axis('off')

axes[1].imshow(gst, cmap='gray')
axes[1].set_title('Smoothed Gradient (T=15% max)')
axes[1].axis('off')

axes[2].imshow(gm, cmap='gray')
axes[2].set_title('Marr-Hildreth (Sigma=3, T=0.002)')
axes[2].axis('off')

axes[3].imshow(gcan, cmap='gray')
axes[3].set_title('Canny (Sigma=2, T=[0.05, 0.15])')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('Figure1026.png')
print("Saved Figure1026.png")
plt.show()
