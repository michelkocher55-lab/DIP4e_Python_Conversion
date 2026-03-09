import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve
import scipy.io as sio

from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e
from libDIP.gaussKernel4e import gaussKernel4e
from libDIPUM.curve_manual_input import curve_manual_input
from libDIPUM.coord2mask import coord2mask
from libDIPUM.data_path import dip_data

# Parameters
n = 21
sig = 5
NIter = 1500
C = 0.5

# Data
img_path = dip_data("breast-implant.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Input initial phi manually.
if os.path.exists("Figure1124.mat"):
    mat = sio.loadmat("Figure1124.mat")
    x = mat["x"].squeeze()
    y = mat["y"].squeeze()
else:
    x, y, vx, vy = curve_manual_input(f, 200, "g.")
    sio.savemat("Figure1124.mat", {"x": x, "y": y, "vx": vx, "vy": vy})

# Create mask for generating initial level set function.
binmask = coord2mask(M, N, x, y)

# Create initial level set function.
phi0 = levelSetFunction4e("mask", binmask)

# Smooth image.
G = gaussKernel4e(n, sig)
fsmooth = convolve(f, G, mode="nearest")

# Compute edge-marking function.
W = levelSetForce4e("gradient", [fsmooth, 1, 50])
T = (np.max(W) + np.min(W)) / 2.0
WBin = W > T

# With reinitialization
phi1 = phi0.copy()
for i in range(1, NIter + 1):
    F = levelSetForce4e("geodesic", [phi1, C, WBin])
    phi1 = levelSetIterate4e(phi1, F)
    if i % 5 == 0:
        phi1 = levelSetReInit4e(phi1, 5, 0.5)

# Without reinitialization
phi2 = phi0.copy()
for _ in range(NIter):
    F = levelSetForce4e("geodesic", [phi2, C, WBin])
    phi2 = levelSetIterate4e(phi2, F)

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(phi0, cmap="gray")
axes[0].axis("off")
axes[1].imshow(phi1, cmap="gray")
axes[1].axis("off")
axes[2].imshow(phi2, cmap="gray")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure1127.png")
plt.show()
