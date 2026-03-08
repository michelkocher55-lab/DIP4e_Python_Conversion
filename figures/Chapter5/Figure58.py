import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from libDIPUM.imnoise2 import imnoise2
from libDIPUM.spfilt import spfilt
from libDIPUM.data_path import dip_data

# Parameters
kernel_size = 3
Q = [1.5, -1.5]
p_salt = 0.05
p_pepper = 0.05

# Data
img_path = dip_data('circuitboard.tif')
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

# Noise adding
fnPepper, _ = imnoise2(f, 'salt & pepper', 0, p_pepper)
fnSalt, _ = imnoise2(f, 'salt & pepper', p_salt, 0)

# Filtering
fHatContraHarmonic1 = spfilt(fnPepper, 'chmean', kernel_size, kernel_size, Q[0])
fHatContraHarmonic2 = spfilt(fnSalt, 'chmean', kernel_size, kernel_size, Q[1])

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

imshow_kwargs = dict(cmap='gray', interpolation='nearest')

axes[0, 0].imshow(fnPepper, **imshow_kwargs)
axes[0, 0].axis('off')
axes[0, 0].set_title('Pepper noise')

axes[0, 1].imshow(fnSalt, **imshow_kwargs)
axes[0, 1].axis('off')
axes[0, 1].set_title('Salt noise')

axes[1, 0].imshow(fHatContraHarmonic1, **imshow_kwargs)
axes[1, 0].axis('off')
axes[1, 0].set_title('Contra harmonic filter')

axes[1, 1].imshow(fHatContraHarmonic2, **imshow_kwargs)
axes[1, 1].axis('off')
axes[1, 1].set_title('Contra harmonic filter')

plt.tight_layout()
plt.savefig('Figure58.png')
plt.show()
