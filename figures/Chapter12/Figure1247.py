"""Figure 12.47 - Corner detection sensitivity and quality level."""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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


print("Running Figure1247...")

# Data
img_path = dip_data('checkerboard-noisy1.tif')
In1 = im2double(plt.imread(img_path))

# Corner detection
C1 = corner(In1)
C2 = corner(In1, SensitivityFactor=0.1)
C3 = corner(In1, SensitivityFactor=0.1, QualityLevel=0.1)
C4 = corner(In1, QualityLevel=0.1)
C5 = corner(In1, QualityLevel=0.3)

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for ax in axes:
    ax.imshow(In1, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

axes[1].plot(C1[:, 0] - 1, C1[:, 1] - 1, "yo", markersize=4)
axes[2].plot(C2[:, 0] - 1, C2[:, 1] - 1, "yo", markersize=4)
axes[3].plot(C3[:, 0] - 1, C3[:, 1] - 1, "yo", markersize=4)
axes[4].plot(C4[:, 0] - 1, C4[:, 1] - 1, "yo", markersize=4)
axes[5].plot(C5[:, 0] - 1, C5[:, 1] - 1, "yo", markersize=4)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1247.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()
