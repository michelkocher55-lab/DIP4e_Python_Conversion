
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.intensityTransformations import intensityTransformations
from libDIPUM.mat2gray import mat2gray
from libDIPUM.data_path import dip_data

# Load and convert to double (0-1 float)
mask = img_as_float(imread(dip_data('angiography-mask-image.tif')))
live = img_as_float(imread(dip_data('angiography-live-image.tif')))

# Difference between mask and live
# d = imsubtract(live,mask);
# Since they are float, we just subtract.
d = live - mask

# d = mat2gray(d);
d = mat2gray(d)

# m = mean2(d);
m = np.mean(d)

# Contrast stretching
# g = intensityTransformations (d,'stretch',m - .03,10);
g = intensityTransformations(d, 'stretch', m - 0.03, 10)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(mask, cmap='gray')
axes[0].set_title('Mask')
axes[0].axis('off')

axes[1].imshow(live, cmap='gray')
axes[1].set_title('Live')
axes[1].axis('off')

axes[2].imshow(d, cmap='gray')
axes[2].set_title('Difference (mat2gray)')
axes[2].axis('off')

axes[3].imshow(g, cmap='gray')
axes[3].set_title('Enhanced (Stretch)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('Figure232.png')
print("Saved Figure232.png")

plt.show()