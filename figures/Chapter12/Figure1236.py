import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.specxture import specxture

print("Running Figure1236 (Radial and angular spectra)...")

# Data
base_dir = "/Users/michelkocher/michel/Data/DIP-DIPUM/DIP"
f1 = imread(os.path.join(base_dir, "matches-random.tif"))
f2 = imread(os.path.join(base_dir, "matches-aligned.tif"))

if f1.ndim == 3:
    f1 = f1[:, :, 0]
if f2.ndim == 3:
    f2 = f2[:, :, 0]

# Spectrum (cartesian mode)
F1 = np.fft.fft2(f1)
F2 = np.fft.fft2(f2)

# Spectrum (polar mode)
F1Rho, F1Theta, _ = specxture(f1)
F2Rho, F2Theta, _ = specxture(f2)

# Display
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0, 0].plot(F1Rho)
ax[0, 0].set_xlabel("rho")
ax[0, 0].set_title("|F_1(rho)|")
ax[0, 0].axis("tight")
# ax[0, 0].set_ylim(0, 9)
ax[0, 0].set_box_aspect(1)

ax[0, 1].plot(F1Theta)
ax[0, 1].set_xlabel("theta")
ax[0, 1].set_title("|F_1(theta)|")
ax[0, 1].axis("tight")
# ax[0, 1].set_ylim(1.8, 2.7)
ax[0, 1].set_box_aspect(1)

ax[1, 0].plot(F2Rho)
ax[1, 0].set_xlabel("rho")
ax[1, 0].set_title("|F_2(rho)|")
ax[1, 0].axis("tight")
# ax[1, 0].set_ylim(0, 6)
ax[1, 0].set_box_aspect(1)

ax[1, 1].plot(F2Theta)
ax[1, 1].set_xlabel("theta")
ax[1, 1].set_title("|F_2(theta)|")
ax[1, 1].axis("tight")
# ax[1, 1].set_ylim(1.6, 3.6)
ax[1, 1].set_box_aspect(1)

fig.tight_layout()
fig.savefig("Figure1236.png")

print("Saved Figure1236.png")
plt.show()
