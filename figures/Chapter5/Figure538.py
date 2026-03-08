import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from General.radon import radon
from libDIPUM.data_path import dip_data

# Parameters
single_theta = 90
# Unused in this figure (kept for parity with MATLAB script)
theta = np.arange(0, 180)
theta2 = [0, 90, 45, 135]
theta3 = np.arange(0, 180, 5.625)
debug = False

# Data
img_path = dip_data('wingding-circle-solid-small.tif')
f1 = img_as_float(imread(img_path))

# Process
R, Rho = radon(f1, single_theta)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(f1, cmap='gray')
axes[0].set_title('f1, r = 100')
axes[0].axis('off')

axes[1].plot(Rho, R, linewidth=1.0)
axes[1].set_xlabel('rho')
axes[1].set_title(f'Radon (f1), theta = {single_theta}')
axes[1].axis('tight')
axes[1].set_aspect('equal', adjustable='box')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('Figure538.png')
plt.show()