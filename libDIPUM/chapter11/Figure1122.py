"""Figure 11.22 - Level set segmentation of breast implant using Eq. (11-96)."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.measure import find_contours
from scipy.ndimage import convolve

from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.gaussKernel4e import gaussKernel4e
from libDIPUM.data_path import dip_data


print("Running Figure1122...")

# Parameters
n = 21
sig = 5
x0 = 370
y0 = 350
r = 18
iterations = [50, 100, 200, 400]

# Data
img_path = dip_data("breast-implant.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Smooth image
G = gaussKernel4e(n, sig)
fsmooth = convolve(f, G, mode="nearest")  # replicate-like boundary

# Initial level set and contour
phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
contours_list = [find_contours(phi0, level=0)]

# Force field + thresholded binary force
F = levelSetForce4e("gradient", [fsmooth, 1, 50])
T = (float(np.min(F)) + float(np.max(F))) / 2.0
FBin = F > T

# Iterate
phi = phi0.copy()
for niter in iterations:
    phi = phi0.copy()
    for _ in range(niter):
        phi = levelSetIterate4e(phi, FBin)
    contours_list.append(find_contours(phi, level=0))

# Final mask from last iterate
X = phi <= 0

# Display Figure 1122
fig1, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].axis("off")

axes[0, 1].imshow(F, cmap="gray")
axes[0, 1].axis("off")

for idx in range(len(iterations)):
    ax = axes.flat[idx + 2]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    for cont in contours_list[idx + 1]:
        ax.plot(cont[:, 1], cont[:, 0], "w.")

# Display Figure 1123
fig2 = plt.figure(2)
plt.imshow(X, cmap="gray")
plt.axis("off")

# Save
out_dir = os.path.dirname(__file__)
out1 = os.path.join(out_dir, "Figure1122.png")
out2 = os.path.join(out_dir, "Figure1123.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
print(f"Saved {out2}")

plt.show()
