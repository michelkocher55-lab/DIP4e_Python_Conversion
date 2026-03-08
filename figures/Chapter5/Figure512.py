import os
import sys
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from libDIPUM.imnoise2 import imnoise2
from libDIPUM.spfilt import spfilt
from libDIPUM.data_path import dip_data

# Parameters
kernel_size = 5
mean = 0
sigma = 0.4
p_salt = 0.05
p_pepper = 0.05

# Data
img_path = dip_data('circuitboard.tif')
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

# Add noise
fnUniform, _ = imnoise2(f, 'uniform', mean, sigma)
fnSaltPepper, _ = imnoise2(f, 'salt & pepper', p_salt, p_pepper)

# Filtering
fnSaltPepperAMean = spfilt(fnSaltPepper, 'amean', kernel_size, kernel_size)
fnSaltPepperGMean = spfilt(fnSaltPepper, 'gmean', kernel_size, kernel_size)
fnSaltPepperMedian = spfilt(fnSaltPepper, 'median', kernel_size, kernel_size)
fnSaltPepperAlphaTrimmed = spfilt(fnSaltPepper, 'atrimmed', kernel_size, kernel_size, 6)

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
imshow_kwargs = dict(cmap='gray', vmin=0, vmax=1, interpolation='nearest')

axes[0, 0].imshow(fnUniform, **imshow_kwargs)
axes[0, 0].set_title('Uniform noise')
axes[0, 0].axis('off')

axes[0, 1].imshow(fnSaltPepper, **imshow_kwargs)
axes[0, 1].set_title('fn = Salt & Pepper noise')
axes[0, 1].axis('off')

axes[0, 2].imshow(fnSaltPepperAMean, **imshow_kwargs)
axes[0, 2].set_title('Arithmetic mean (fn)')
axes[0, 2].axis('off')

axes[1, 0].imshow(fnSaltPepperGMean, **imshow_kwargs)
axes[1, 0].set_title('Geometric mean (fn)')
axes[1, 0].axis('off')

axes[1, 1].imshow(fnSaltPepperMedian, **imshow_kwargs)
axes[1, 1].set_title('Median (fn)')
axes[1, 1].axis('off')

axes[1, 2].imshow(fnSaltPepperAlphaTrimmed, **imshow_kwargs)
axes[1, 2].set_title('Alpha trimmed mean (fn)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('Figure512.png')
plt.show()