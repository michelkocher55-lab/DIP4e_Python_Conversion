
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, resize

# Parameters
NR = 256
# Theta = 0 : 1 : 179;
Theta = np.arange(0, 180, 1)

# Data
X = np.zeros((NR, NR))
r_start = int(NR/4)
r_end = int(3*NR/4)
c_center = int(NR/2)
c_start = c_center - 20
c_end = c_center + 20
X[r_start:r_end, c_start:c_end] = 1

# Radon transform
R = radon(X, theta=Theta, circle=False)

# Inverse Radon transform
print("Computing Inverse Radon (Ram-Lak)...")
XHat_RamLak = iradon(R, theta=Theta, filter_name='ramp', circle=False)

# XHat.Hamming = iradon (R, Theta, 'Hamming');
print("Computing Inverse Radon (Hamming)...")
XHat_Hamming = iradon(R, theta=Theta, filter_name='hamming', circle=False)

if XHat_RamLak.shape[0] > NR:
    # Center crop
    diff = XHat_RamLak.shape[0] - NR
    start = diff // 2
    XHat_RamLak = XHat_RamLak[start:start+NR, start:start+NR]

if XHat_Hamming.shape[0] > NR:
    diff = XHat_Hamming.shape[0] - NR
    start = diff // 2
    XHat_Hamming = XHat_Hamming[start:start+NR, start:start+NR]

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Original
axes[0, 0].imshow(X, cmap='gray')
axes[0, 0].set_title('Orig')
axes[0, 0].axis('off')

# 2. Radon
# Display sinogram. Transpose to match (theta (y), rho (x)) or typical MATLAB view?
# MATLAB: imshow(R, []).
# If R is (rho, theta) in Python (skimage default), then imshow shows rho on y-axis (rows), theta on x-axis (cols).
axes[0, 1].imshow(R, cmap='gray', aspect='auto')
axes[0, 1].set_title('Radon transform')
axes[0, 1].set_xlabel('theta')
axes[0, 1].set_ylabel('rho')

# 3. Ram-Lak
axes[1, 0].imshow(XHat_RamLak, cmap='gray')
axes[1, 0].set_title('RAM-LAK')
axes[1, 0].axis('off')

# 4. Hamming
axes[1, 1].imshow(XHat_Hamming, cmap='gray')
axes[1, 1].set_title('RAM-LAK + Hamming')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure543.png')
print("Saved Figure543.png")
plt.show()