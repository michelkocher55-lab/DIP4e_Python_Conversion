"""Figure 12.60 - Keypoints without orientation arrows."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from libDIP.boundary2image4e import boundary2image4e
from libDIPUM.sift import sift
import ia870 as ia
from libDIPUM.data_path import dip_data


print("Running Figure1260New (keypoints without orientation arrows)...")

# Data
img_path_pgm = dip_data('building-600by600.pgm')

# SIFT
image, descrips, locs = sift(img_path_pgm)

# Keep the same fixed-index behavior, clipped to available keypoints.
k = min(643, locs.shape[0])
rows = locs[:k, 0]
cols = locs[:k, 1]
b = np.column_stack((rows, cols))

keypoints = boundary2image4e(b, 600, 600)

# Add 4-neighbors to enlarge each keypoint.
keypoints_Enlarged = ia.iadil(keypoints, ia.iasecross(2))

# Display
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(keypoints, cmap="gray")
axs[0].set_title(f"Keypoints (k = {k})")
axs[0].axis("off")

axs[1].imshow(keypoints_Enlarged, cmap="gray")
axs[1].set_title(f"Enlarged keypoints (k = {k})")
axs[1].axis("off")

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1260.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")

plt.show()
