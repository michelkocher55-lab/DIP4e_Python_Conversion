
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import ia870 as ia
from libDIPUM.data_path import dip_data

# Load images
forig = imread(dip_data('chronometer-2136x2140-2pt3-inch-930-dpi.tif'))
f300 = imread(dip_data('chronometer-689x690-2pt3-inch-300-dpi.tif'))
f150 = imread(dip_data('chronometer-345x345-2pt3-inch-150-dpi.tif'))
f72 = imread(dip_data('chronometer-165x166-2pt3-inch-72-dpi.tif'))

# Differences
# Typically resizing differences require images to be same size.
# The filenames indicate different sizes (e.g. 165x166).
# Figure 2.31 description: "Differences between original chronometer ... and ... versions from Fig 2.23".
# But usually subtraction requires same dimensions.
# Figure 2.23 resizes them back to original size.
# The files on disk might be the *resized* versions?
# Or are they the *re-zoomed* versions?
# The files "f300" (689x690) seem to be the downsampled ones.
# "forig" is 2136x2140.
# If we just subtract, it fails.
# The script `Figure231.m` creates `d300 = imsubtract(forig, f300)`.
# This implies f300 MUST be same size as forig in MATLAB workspace context.
# BUT `Figure223.m` (referenced) does the zooming back.
# The files on disk `Fig0223(b)... 300 dpi` likely refer to the *result* of 2.23?
# Or maybe the intermediate reduced ones?
# Filename says `689x690` which is smaller.
# So we probably need to resize them back to `forig.shape` before subtraction?
# Or maybe the files on disk ARE the zoomed ones but misnamed?
# Let's check size of loaded images.


# Calculate difference (absolute difference usually? or signed clipped?)
# imsubtract in MATLAB: P - Q. saturated.
# If we want to visualize "pixelation artifacts", usually we take diff.
# Let's do abs diff or saturation?
# Script says `imsubtract`. So `forig - f300`. any negative becomes 0.

# Convert to float for safe subtract then clip?
# MATLAB `imsubtract` on uint8 clips negatives to 0.

d300 = np.clip(forig.astype(float) - f300.astype(float), 0, 255).astype(np.uint8)
d150 = np.clip(forig.astype(float) - f150.astype(float), 0, 255).astype(np.uint8)
d72 = np.clip(forig.astype(float) - f72.astype(float), 0, 255).astype(np.uint8)

# Dilate
B = ia.iasebox (2);

gd72 = ia.iadil(d72, B)
gd150 = ia.iadil(d150, B)
gd300 = ia.iadil(d300, B)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

axes[0].imshow(gd72, cmap='gray')
axes[0].set_title('Dilated Diff (72 dpi)')
axes[0].axis('off')

axes[1].imshow(gd150, cmap='gray')
axes[1].set_title('Dilated Diff (150 dpi)')
axes[1].axis('off')

axes[2].imshow(gd300, cmap='gray')
axes[2].set_title('Dilated Diff (300 dpi)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure231.png')
print("Saved Figure231.png")

plt.show()