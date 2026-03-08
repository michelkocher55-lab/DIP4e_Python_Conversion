"""Figure 12.50 - Rotated building with Harris corners."""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from General.corner import corner
from libDIPUM.data_path import dip_data


def im2double(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        if info.min < 0:
            return (a.astype(np.float64) - info.min) / (info.max - info.min)
        return a.astype(np.float64) / info.max
    if a.dtype == np.bool_:
        return a.astype(np.float64)
    return a.astype(np.float64)


print("Running Figure1250...")

# Data
img_path = dip_data('building-600by600.tif')
IB = im2double(plt.imread(img_path))

# Rotation (uncropped / loose)
IBR = ndimage.rotate(IB, 5.0, reshape=True, order=1, mode="constant", cval=0.0)

# Cropping to remove black borders introduced by rotation.
# MATLAB indices: IBR(55:596,53:591)
IBRc = IBR[54:596, 52:591]

# Corners with same settings as Figure 12.49 last panel.
C = corner(IBRc, QualityLevel=0.07)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax in axes:
    ax.imshow(IBRc, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

axes[1].plot(C[:, 0] - 1, C[:, 1] - 1, "yo", markersize=4)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1250.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()

