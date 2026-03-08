
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from libDIP.checkerimage import checkerimage

print("Running Figure418 (Aliasing with Checkerboard)...")

# Parameters
M = 96

# Generate images
# g1=checkerimage(0.0625,96);
g1 = checkerimage(0.0625, M)

# g2=checkerimage(0.1667,96);
g2 = checkerimage(0.1667, M)

# g3=checkerimage(1.05,96);
g3 = checkerimage(1.05, M)

# g4=checkerimage(2.084,96);
g4 = checkerimage(2.084, M)

# Save images (using 'tif' as in MATLAB script, or 'png' for easy viewing?
# MATLAB: imwrite(..., 'Fig4.18...tif')
# I will save as .tif to match, but also display/save a composite png.

# Note: Images are 0/1 binary. imsave expects appropriate type or converts.
# We should convert to uint8 (0, 255) for standard TIFF compatibility.

g1_u8 = (g1 * 255).astype(np.uint8)
g2_u8 = (g2 * 255).astype(np.uint8)
g3_u8 = (g3 * 255).astype(np.uint8)
g4_u8 = (g4 * 255).astype(np.uint8)

# Display composite
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes = axes.flatten()

axes[0].imshow(g1, cmap='gray')
axes[0].set_title('delX = 0.0625 (No aliasing)')
axes[0].axis('off')

axes[1].imshow(g2, cmap='gray')
axes[1].set_title('delX = 0.1667 (No aliasing)')
axes[1].axis('off')

axes[2].imshow(g3, cmap='gray')
axes[2].set_title('delX = 1.05 (Aliasing)')
axes[2].axis('off')

axes[3].imshow(g4, cmap='gray')
axes[3].set_title('delX = 2.084 (Severe Aliasing)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('Figure418.png')
print("Saved Figure418.png and component TIFs.")
plt.show()