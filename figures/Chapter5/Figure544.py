
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize

# Parameters
NR = 256
# Theta = 0 : 1 : 179;
Theta = np.arange(0, 180, 1)

# Data
base_phantom = shepp_logan_phantom()
f = resize(base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True)

# Radon transform
R = radon(f, theta=Theta, circle=False)

# Inverse Radon transform
print("Computing Inverse Radon (Ram-Lak)...")
fHat_RamLak = iradon(R, theta=Theta, filter_name='ramp', circle=False)

# fHat.Hamming = iradon (R, Theta, 'Hamming');
print("Computing Inverse Radon (Hamming)...")
fHat_Hamming = iradon(R, theta=Theta, filter_name='hamming', circle=False)

# Crop to size if necessary
if fHat_RamLak.shape[0] > NR:
    diff = fHat_RamLak.shape[0] - NR
    start = diff // 2
    fHat_RamLak = fHat_RamLak[start:start+NR, start:start+NR]

if fHat_Hamming.shape[0] > NR:
    diff = fHat_Hamming.shape[0] - NR
    start = diff // 2
    fHat_Hamming = fHat_Hamming[start:start+NR, start:start+NR]

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Original
axes[0, 0].imshow(f, cmap='gray')
axes[0, 0].set_title('Orig')
axes[0, 0].axis('off')

# 2. Radon
axes[0, 1].imshow(R, cmap='gray', aspect='auto')
axes[0, 1].set_title('Radon transform')
axes[0, 1].set_xlabel('theta')
axes[0, 1].set_ylabel('rho')

# 3. Ram-Lak
axes[1, 0].imshow(fHat_RamLak, cmap='gray')
axes[1, 0].set_title('RAM-LAK')
axes[1, 0].axis('off')

# 4. Hamming
axes[1, 1].imshow(fHat_Hamming, cmap='gray')
axes[1, 1].set_title('RAM-LAK + Hamming')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure544.png')
print("Saved Figure544.png")
plt.show()