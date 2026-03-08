"""Figure 12.63 - Matching of building corner."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt

from libDIPUM.sift import sift
from libDIPUM.showkeys import showkeys
from libDIPUM.match import match
from libDIPUM.data_path import dip_data

print("Running Figure1263 (matching of building corner)...")

# Data
img1 = dip_data('building-600by600.pgm')
img2 = dip_data('building-corner.pgm')

# Keypoints for building
image, descrips, locs1 = sift(img1)
overlay1 = showkeys(image, locs1)
print(f"Building keypoints found: {locs1.shape[0]}")

# Keypoints for building corner
image, descrips, locs2 = sift(img2)
overlay2 = showkeys(image, locs2)
print(f"Building-corner keypoints found: {locs2.shape[0]}")

# Match between the two
num, overlay_match = match(img1, img2)
print(f"Found {num} matches")

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

# Save (equivalent to MATLAB print -f1/-f2/-f3)
out_dir = os.path.dirname(__file__)
out1 = os.path.join(out_dir, "Figure1263.png")
out2 = os.path.join(out_dir, "Figure1263Bis.png")
out3 = os.path.join(out_dir, "Figure1263Ter.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
print(f"Saved {out2}")
print(f"Saved {out3}")

plt.show()
