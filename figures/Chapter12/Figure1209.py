import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.im2minperpoly import im2minperpoly
from libDIPUM.connectpoly import connectpoly
from libDIPUM.bound2im import bound2im
from libDIPUM.bwboundaries import bwboundaries
from libDIPUM.data_path import dip_data

print("Running Figure1209 (Minimum Perimeter Polygon)...")

# 1. Data
path = dip_data("mapleleaf.tif")
B = imread(path)
if B.ndim == 3:
    B = B[:, :, 0]
B = B > 0  # Assume >0 is object

LesCellSize = [2, 4, 6, 8, 16, 32]
M, N = B.shape

# Boundaries
boundaries = bwboundaries(B, conn=8)
if boundaries:
    b = boundaries[0]
else:
    b = np.array([])

bIm = bound2im(b, M, N)

B2_list = []
B2_coords = []
LesX_list = []

for cellsize in LesCellSize:
    print(f"Processing Cell Size {cellsize}...")
    X, Y, R = im2minperpoly(B, cellsize)
    LesX_list.append(len(X))

    # ConnectPoly
    if len(X) > 0:
        b2 = connectpoly(X, Y)
        B2 = bound2im(b2, M, N)
        B2_coords.append(b2)
    else:
        B2 = np.zeros_like(B)
        B2_coords.append(np.zeros((0, 2), dtype=int))

    B2_list.append(B2)

# Display
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
ax = axes.ravel()

# 1. Original
ax[0].imshow(B, cmap="gray")
ax[0].set_title(f"X, size = {B.shape}")
ax[0].axis("off")

# 2. Boundary (negative display with dotted boundary points)
ax[1].imshow(np.ones_like(B), cmap="gray", vmin=0, vmax=1)
if len(b) > 0:
    ax[1].plot(b[:, 1], b[:, 0], "k.", markersize=1.0)
ax[1].set_title(f"8 conn., N_Ver = {len(b)}")
ax[1].set_aspect("equal")
ax[1].axis("off")

for i, cellsize in enumerate(LesCellSize):
    idx = i + 2
    if idx < 8:
        pts = B2_coords[i]
        # Overlay approximated contour on original image.
        ax[idx].imshow(B, cmap="gray")
        if len(pts) > 0:
            ax[idx].plot(pts[:, 1], pts[:, 0], "r.", markersize=1.1)
        ax[idx].set_title(f"CS={cellsize}, N_Ver={LesX_list[i]}")
        ax[idx].set_aspect("equal")
        ax[idx].axis("off")

plt.tight_layout()
plt.savefig("Figure1209.png")
plt.show()
