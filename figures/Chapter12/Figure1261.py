"""Figure 12.61 - Keypoints with orientation."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt

from libDIPUM.sift import sift
from libDIPUM.showkeys import showkeys
from libDIPUM.data_path import dip_data

print("Running Figure1261 (keypoints with orientation)...")

# Data
img_path_pgm = dip_data('building-600by600.pgm')

# SIFT
image, descrips, locs = sift(img_path_pgm)
print(f"Detected keypoints: {locs.shape[0]}")

# Display
overlay = showkeys(image, locs)
plt.figure()
plt.imshow(overlay)
plt.title("Keypoints with orientation")
plt.axis("off")

out_path = os.path.join(os.path.dirname(__file__), "Figure1261.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()
