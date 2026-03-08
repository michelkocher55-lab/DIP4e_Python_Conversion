import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve

from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.gaussKernel4e import gaussKernel4e
from libDIPUM.data_path import dip_data

# Parameters
n = 21
sig = 5

# Data
img_path = dip_data('breast-implant.tif')
f = img_as_float(imread(img_path))
M, N = f.shape

# Smooth image (fspecial('gaussian') + imfilter('replicate'))
G = gaussKernel4e(n, sig)
fsmooth = convolve(f, G, mode='nearest')

# Compute edge-marking function
W = levelSetForce4e('gradient', [fsmooth, 1, 50])
T = (np.max(W) + np.min(W)) / 2.0
WBin = W > T

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(fsmooth, cmap='gray')
axes[0].axis('off')
axes[1].imshow(W, cmap='gray')
axes[1].axis('off')
axes[2].imshow(WBin, cmap='gray')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure1125.png')
plt.show()
