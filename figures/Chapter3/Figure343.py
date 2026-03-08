
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

# Kernels
# gauss43 = gaussiankernel(43,'sampled',7,1); % Approx 6 sig
# gauss85 = gaussiankernel(85,'sampled',7,1); % Approx 12 sig

gauss43, _ = gaussiankernel(43, 'sampled', 7.0, 1.0)
gauss85, _ = gaussiankernel(85, 'sampled', 7.0, 1.0)

# Normalize the filters
gauss43 = gauss43 / np.sum(gauss43)
gauss85 = gauss85 / np.sum(gauss85)

# Filter (default padding is zero padding)
ggauss43 = correlate(f, gauss43, mode='constant', cval=0.0)
ggauss85 = correlate(f, gauss85, mode='constant', cval=0.0)

# Compare
# diff = imsubtract(ggauss43, ggauss85);
# Since images are floats, simple subtraction.
diff = ggauss43 - ggauss85
# MATLAB imsubtract on doubles clip negatives?
# "Z = imsubtract(X,Y) subtracts each element ...
# If the output array is of integer class, then all negative results are truncated to zero."
# If double, negative values are preserved.
# HOWEVER, the script later computes `255 * max(diff(:))`.
# Wait, usually for display one might care about abs difference, but `imsubtract` on double is just minus.
# The title implies we look at the deviation.
# I will stick to simple subtraction as f is double.

max_diff = 255 * np.max(diff)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(ggauss43, cmap='gray')
axes[1].set_title('Gaussian 43x43')
axes[1].axis('off')

axes[2].imshow(ggauss85, cmap='gray')
axes[2].set_title('Gaussian 85x85')
axes[2].axis('off')

# Subplot 4: diff
axes[3].imshow(diff, cmap='gray')
axes[3].set_title(f'Diff = {max_diff:.4f}')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('Figure343.png')
print("Saved Figure343.png")
plt.show()