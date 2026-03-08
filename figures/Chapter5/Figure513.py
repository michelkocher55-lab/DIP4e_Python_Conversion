import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from scipy.ndimage import uniform_filter
from libDIPUM.imnoise2 import imnoise2
from libDIPUM.spfilt import spfilt
from libDIPUM.data_path import dip_data

def adaptive_noise_reduction_filter(f, w, noise_var):
    """
    Adaptive noise reduction filter (MATLAB-equivalent).
    w: half-window size (window is (2*w+1)x(2*w+1))
    noise_var: noise variance
    """
    f = img_as_float(f)
    size = 2 * w + 1

    # Local mean and variance (population variance, like MATLAB var(...,1))
    local_mean = uniform_filter(f, size=size, mode='reflect')
    local_mean_sq = uniform_filter(f * f, size=size, mode='reflect')
    local_var = local_mean_sq - local_mean ** 2

    # Apply adaptive formula
    with np.errstate(divide='ignore', invalid='ignore'):
        g = f - (noise_var / local_var) * (f - local_mean)

    # Where local variance is zero, keep original pixel
    g = np.where(local_var > 0, g, f)
    return g


# Parameters
kernel_size = 7
mean = 0
sigma = 0.1

# Data
img_path = dip_data('circuitboard.tif')
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

# Add noise
fnGaussian, _ = imnoise2(f, 'gaussian', mean, sigma)

# Filtering
fnGaussianAMean = spfilt(fnGaussian, 'amean', kernel_size, kernel_size)
fnGaussianGMean = spfilt(fnGaussian, 'gmean', kernel_size, kernel_size)
fnGaussianAdaptiveFilter = adaptive_noise_reduction_filter(fnGaussian, 3, sigma ** 2)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
imshow_kwargs = dict(cmap='gray', vmin=0, vmax=1, interpolation='nearest')

axes[0, 0].imshow(fnGaussian, **imshow_kwargs)
axes[0, 0].set_title('Gaussian noise')
axes[0, 0].axis('off')

axes[0, 1].imshow(fnGaussianAMean, **imshow_kwargs)
axes[0, 1].set_title('Arithmetic mean')
axes[0, 1].axis('off')

axes[1, 0].imshow(fnGaussianGMean, **imshow_kwargs)
axes[1, 0].set_title('Geometric mean')
axes[1, 0].axis('off')

axes[1, 1].imshow(fnGaussianAdaptiveFilter, **imshow_kwargs)
axes[1, 1].set_title('Adaptive Filter (fn)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure513.png')
plt.show()
