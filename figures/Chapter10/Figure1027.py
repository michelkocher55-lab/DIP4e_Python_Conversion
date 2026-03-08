
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.morphology import thin
from libDIPUM.edgelinklocal import edgelinklocal
from libDIPUM.data_path import dip_data

print("Running Figure1027 (Local Edge Linking)...")

# Data
# Filename in script is 'van-rear.tif'.
# I'll rely on the finding tool or standard paths.

img_path = dip_data('van-rear.tif')

f = img_as_float(imread(img_path))

# Detection and linking in the vertical direction:
# [GV, MAGV, ANGLEV, Gxv, Gyv] = edgelinklocal(f', .30, 90, 45, 25);
# Transpose input f
ft = f.T
GV_t, MAGV_t, ANGLEV_t, Gxv_t, Gyv_t = edgelinklocal(ft, 0.30, 90, 45, 25)

# Transpose back
GV = GV_t.T
MAGV = MAGV_t.T
# ANGLEV = ANGLEV_t.T
# Gxv = Gxv_t.T
# Gyv = Gyv_t.T

# Detection and linking in the horizontal direction:
# [GH, MAGH, ANGLEH, Gxh, Gyh] = edgelinklocal(f, .30, 90, 45, 25);
GH, MAGH, ANGLEH, Gxh, Gyh = edgelinklocal(f, 0.30, 90, 45, 25)

# Logical OR
# G = GH | GV;
G = (GH > 0) | (GV > 0)

# Thinning
# Gthin = bwmorph (G,'thin',Inf);
# skimage.morphology.thin returns bool
Gthin = thin(G)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(MAGH, cmap='gray')
axes[1].set_title('Gradient Magnitude') # MAGH ~ MAGV
axes[1].axis('off')

axes[2].imshow(GH, cmap='gray')
axes[2].set_title('Horizontal Linking')
axes[2].axis('off')

axes[3].imshow(GV, cmap='gray')
axes[3].set_title('Vertical Linking')
axes[3].axis('off')

axes[4].imshow(G, cmap='gray')
axes[4].set_title('Combined (OR)')
axes[4].axis('off')

axes[5].imshow(Gthin, cmap='gray')
axes[5].set_title('Thinned Result')
axes[5].axis('off')

plt.tight_layout()
plt.savefig('Figure1027.png')
print("Saved Figure1027.png")
plt.show()