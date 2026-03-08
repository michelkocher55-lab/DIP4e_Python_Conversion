
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.io import imread
from PIL import Image
from libDIPUM.intensityTransformations import intensityTransformations
from libDIPUM.data_path import dip_data


# Image loading
img_name = dip_data('pollen-lowcontrast.tif')
f = imread(img_name)
if f.ndim == 3: f = f[:,:,0]

# TXFun = [0, 1/8, 7/8, 1]; % For input = 0, 1/3, 2/3, 1
TXFun = np.array([0, 1/8, 7/8, 1])

# Contrast stretching
# f needs to be float [0,1] or intensityTransformations handles it?
# intensityTransformations handles conversion to float then applies specified.
g = intensityTransformations(f, 'specified', TXFun)

# Otsu threshold
# skimage.filters.threshold_otsu needs input in certain range?
# g is float [0,1] from specified transform (if inputs were standard).
# intensityTransformations returns float for 'specified'.

thresh = filters.threshold_otsu(g)
X = g > thresh

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Plot Law
# axis inputs: 0, 1/3, 2/3, 1
x_law = np.linspace(0, 1, 4)
axes[0, 0].plot(x_law, TXFun, 'b-o')
axes[0, 0].set_title('Contrast stretching law')
axes[0, 0].set_xlim([0, 1])
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(True)

# 2. Input
axes[0, 1].imshow(f, cmap='gray',vmin=0, vmax=255)
axes[0, 1].set_title('Input image')
axes[0, 1].axis('off')

# 3. Output
axes[1, 0].imshow(g, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('Output image')
axes[1, 0].axis('off')

# 4. Binary
axes[1, 1].imshow(X, cmap='gray')
axes[1, 1].set_title('Otsu applied to output image')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure310.png')
print("Saved Figure310.png")
plt.show()