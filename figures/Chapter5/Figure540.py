
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from libDIP.imRecon4e import imRecon4e
from libDIP.intScaling4e import intScaling4e

# Parameters
NR = 256
# Theta = 0 : 1 : 179;
Theta = np.arange(0, 180, 1)

# Data
# Rectangle
Rectangle = np.zeros((NR, NR))
r_start = int(NR/4)
r_end = int(3*NR/4)
c_center = int(NR/2)
c_start = c_center - 20
c_end = c_center + 20

Rectangle[r_start:r_end, c_start:c_end] = 1

# SheppLogan
base_phantom = shepp_logan_phantom()
SheppLogan = resize(base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True)

# Display
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 1. Rectangle BackProj
print("Computing BackProj for Rectangle...")
rec_rect = imRecon4e(Rectangle, Theta)
axes[0].imshow(rec_rect, cmap='gray')
axes[0].set_title('Back Projection 180 angles (Rect)')
axes[0].axis('off')

# 2. SheppLogan BackProj
print("Computing BackProj for SheppLogan...")
rec_shepp = imRecon4e(SheppLogan, Theta)
axes[1].imshow(rec_shepp, cmap='gray')
axes[1].set_title('Back Projection 180 angles (Shepp)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Figure540.png')
print("Saved Figure540.png")
plt.show()