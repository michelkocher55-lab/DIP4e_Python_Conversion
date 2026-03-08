
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import correlate
from libDIPUM.gaussiankernel import gaussiankernel
from libDIPUM.data_path import dip_data

# Image loading
img_name = dip_data('testpattern1024.tif')
f = imread(img_name)
if f.ndim == 3: f = f[:,:,0]

f = img_as_float(f)

# 21x21 Gaussian kernel. sig = 3.5
# MATLAB: gaussiankernel(21,'sampled',3.5,1);
# Python: returns (w, S)
# The last argument '1' in MATLAB likely is 'K'.
# My python implementation: args are (sigma, K).
gauss3pt5, S = gaussiankernel(21, 'sampled', 3.5, 1.0)

# Normalize
# gauss3pt5 = gauss3pt5/sum(gauss3pt5(:))
gauss3pt5 = gauss3pt5 / np.sum(gauss3pt5)

# Filter. Default padding is zero padding.
# MATLAB imfilter corresponds to correlation.
# scipy.ndimage.correlate with mode='constant', cval=0.0 corresponds to zero padding.
ggauss3pt5 = correlate(f, gauss3pt5, mode='constant', cval=0.0)

# sig = 7, size 43x43
gauss7, S2 = gaussiankernel(43, 'sampled', 7.0, 1.0)
gauss7 = gauss7 / np.sum(gauss7)

ggauss7 = correlate(f, gauss7, mode='constant', cval=0.0)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(ggauss3pt5, cmap='gray')
axes[1].set_title('Gaussian 21x21, sigma=3.5')
axes[1].axis('off')

axes[2].imshow(ggauss7, cmap='gray')
axes[2].set_title('Gaussian 43x43, sigma=7')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure342.png')
print("Saved Figure342.png")
plt.show()