
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import exposure
from libDIPUM.data_path import dip_data

# Image loading
img_name = dip_data('hidden-symbols.tif')

f = imread(img_name)
if f.ndim == 3: f = f[:, :, 0]

NR, NC = f.shape
Local = 3

# Global HE
# skimage histeq returns float
g1_float = exposure.equalize_hist(f)
g1 = (g1_float * 255).astype(np.uint8)

# Local HE
# MATLAB: NumTiles = [floor(NR/Local), floor(NC/Local)]
# If Local=3, Tile Size is approx 3x3.
# skimage equalize_adapthist takes kernel_size.
# kernel_size = [Local, Local] (integer dimensions of the tile/window).

# ClipLimit = 1.
# In skimage, clip_limit is normalized between 0 and 1.
# MATLAB 'ClipLimit' is also 0 to 1.
# So we use clip_limit=1.0.

# Note: equalize_adapthist acts on float [0,1] image usually or converts it.
# It returns float [0,1].

# tile_size calculation:
# MATLAB: NumTiles.
# skimage: kernel_size.
# If NumTiles = NR/3. Then pixel size of tile is NR/(NR/3) = 3.
# So kernel_size = (3, 3).

# However, skimage's kernel_size corresponds to the window size for sliding window?
# No, equalize_adapthist uses CLAHE which operates on tiles.
# "kernel_size: integer or list-like, optional. Defines the shape of contextual regions used in the algorithm."
# Default is image_shape/8.

# Implementation:
g2_float = exposure.equalize_adapthist(f, kernel_size=(Local, Local), clip_limit=1.0)
g2 = (g2_float * 255).astype(np.uint8)

# Display
fig, axes = plt.subplots(1, 3, figsize=(10, 10))
axes = axes.flatten()

# 1. Original
axes[0].imshow(f, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original')
axes[0].axis('off')

# 2. Global HE
axes[1].imshow(g1, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Global hist eq.')
axes[1].axis('off')

# 3. Local HE
axes[2].imshow(g2, cmap='gray', vmin=0, vmax=255)
axes[2].set_title(f'Local Size = {Local}')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure332.png')
print("Saved Figure332.png")
plt.show()