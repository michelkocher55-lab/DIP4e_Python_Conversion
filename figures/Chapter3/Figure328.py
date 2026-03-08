
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import equalize_hist, histogram
from libDIPUM.spechist import spechist
from libDIPUM.fun2hist import fun2hist
from libDIPUM.trapezmf import trapezmf
from libDIPUM.data_path import dip_data

# Image loading
img_path = dip_data('hidden-horse.tif')
f = imread(img_path)
if f.ndim == 3: f = f[:,:,0]

M, N = f.shape

# Construct a non-normalized uniform histogram
# z = 1:256
z = np.arange(1, 257)

# fun = trapezmf(z, 0, 0, 256, 256)
# This creates a uniform distributions of 1s.
tmf = trapezmf(z, 0, 0, 256, 256)

# H = fun2hist(fun, M*N)
H = fun2hist(tmf, M * N)

# "Normal" histogram equalized image
# histeq in MATLAB returns [0, 255]? Or [0, 1]?
# In skimage, equalize_hist returns [0, 1].
geq_norm = equalize_hist(f, nbins=256)
geq = (geq_norm * 255).astype(np.uint8)

# Histogram specified image
gsp = spechist(f, H)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

# 1. Original (Not explicitly shown in MATLAB script subplot logic which seems slightly off, but typical)
# MATLAB script writes: subplot(2,2,1); imshow(g).
# But 'g' corresponds to spechist output in standard notation, OR original 'f'?
# Given the script variable names: f (input), geq (histeq), gsp (spechist).
# If variable 'g' is not defined, it might be a typo in user's provided file or referring to previous workspace var.
# I will assume subplot 1 should show the Original image 'f'.
axes[0].imshow(f, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original')
axes[0].axis('off')

# 2. Hist Eq
axes[1].imshow(geq, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Hist Eq')
axes[1].axis('off')

# 3. Specified
axes[2].imshow(gsp, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Hist Specified')
axes[2].axis('off')

# 4. Histogram of Specified
counts, centers = histogram(gsp, nbins=256, source_range='image')
axes[3].bar(centers, counts, width=1)
axes[3].set_title('Histogram of Specified')
axes[3].set_xlim([0, 255])

plt.tight_layout()
plt.savefig('Figure328.png')
print("Saved Figure328.png")
plt.show()