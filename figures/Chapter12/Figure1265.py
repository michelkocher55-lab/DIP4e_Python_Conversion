"""Figure 12.65 - Matching of half-size building corner."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from scipy import ndimage

from libDIPUM.sift import sift
from libDIPUM.showkeys import showkeys
from libDIPUM.match import match
from libDIPUM.data_path import dip_data

print("Running Figure1265 (matching of half-size building corner)...")

# Parameters
ScaleFactor = 0.5

# Data (rotated image) - kept for parity with MATLAB script
I = iio.imread(dip_data('building-600by600.tif'))
if I.ndim == 2:
    Ih = ndimage.zoom(I, ScaleFactor, order=1)
else:
    Ih = ndimage.zoom(I, (ScaleFactor, ScaleFactor, 1), order=1)
_ = Ih  # variable intentionally retained to mirror MATLAB flow

# Keypoints for half-size building
img1 = dip_data('building-halfsize.pgm')
image, descrips, locs1 = sift(img1)
overlay1 = showkeys(image, locs1)

# Keypoints for half-size building corner
img2 = dip_data('building-halfsize-corner.pgm')
image, descrips, locs2 = sift(img2)
overlay2 = showkeys(image, locs2)

# Match between the two
num, overlay_match = match(img1, img2)

# Display
fig1 = plt.figure(1)
plt.imshow(overlay1)
plt.title(f"Building keypoints, {locs1.shape[0]}, keypoints")
plt.axis("off")

fig2 = plt.figure(2)
plt.imshow(overlay2)
plt.title(f"Building-corner keypoints, {locs2.shape[0]}, keypoints")
plt.axis("off")

fig3 = plt.figure(3)
plt.imshow(overlay_match)
plt.title(f"Matching, {num}, match")
plt.axis("off")

# Save
out_dir = os.path.dirname(__file__)
out1 = os.path.join(out_dir, "Figure1265.png")
out2 = os.path.join(out_dir, "Figure1265Bis.png")
out3 = os.path.join(out_dir, "Figure1265Ter.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
print(f"Saved {out2}")
print(f"Saved {out3}")

plt.show()
