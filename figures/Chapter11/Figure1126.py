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
from libDIP.gaussKernel4e import gaussKernel4e
from libDIPUM.curve_manual_input import curve_manual_input
from libDIPUM.coord2mask import coord2mask
from libDIPUM.curve_display import curve_display
from libDIPUM.contourc import contourc
from libDIPUM.data_path import dip_data

# Input initial phi manually.
if os.path.exists("Figure1124.mat"):
    mat = sio.loadmat("Figure1124.mat")
    x = mat["y"].squeeze()
    y = mat["x"].squeeze()
else:
    # Data is needed for manual input display
    img_path = dip_data("breast-implant.tif")
    f_tmp = img_as_float(imread(img_path))
    x, y, vx, vy = curve_manual_input(f_tmp, 200, "g.")
    sio.savemat("Figure1124.mat", {"x": x, "y": y, "vx": vx, "vy": vy})

# Parameters
n = 21
sig = 5
iterations = [1500]

# Data
img_path = dip_data("breast-implant.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Create mask for generating initial level set function.
binmask = coord2mask(M, N, x, y)

# Create initial level set function.
phi0 = levelSetFunction4e("mask", binmask)

# Obtain the zero-level set contour.
c_list = [contourc(phi0, [0, 0])]

# Smooth image.
G = gaussKernel4e(n, sig)
fsmooth = convolve(f, G, mode="nearest")

# Compute edge-marking function.
W = levelSetForce4e("gradient", [fsmooth, 1, 50])
T = (np.max(W) + np.min(W)) / 2.0
WBin = W > T

# Iterate.
for niter in iterations:
    phi = phi0.copy()
    C = 0.5
    for _ in range(niter):
        F = levelSetForce4e("geodesic", [phi, C, WBin])
        phi = levelSetIterate4e(phi, F)
    c_list.append(contourc(phi, [0, 0]))

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for idx, cc in enumerate(c_list):
    ax = axes[idx]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    curve_display(cc[1, :], cc[0, :], "y.", ax=ax)

plt.tight_layout()
plt.savefig("Figure1126.png")
plt.show()
