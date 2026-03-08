import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.homomorphictf import homomorphictf
from libDIP.dftFiltering4e import dftFiltering4e
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Parameters
GammaH = 3
GammaL = 0.4
c = 5
D0 = 20

# Data
img_path = dip_data('PET-scan.tif')
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Homomorphic filter transfer function.
H = homomorphictf(P, Q, GammaL, GammaH, c, D0)

# H is not centered. Center it.
H = np.fft.fftshift(H)

# Use H to filter f.
g = dftFiltering4e(f, H)
g = intScaling4e(g)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f, cmap="gray")
axes[0].axis("off")

axes[1].imshow(g, cmap="gray")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure460.png", dpi=300, bbox_inches="tight")
plt.show()
