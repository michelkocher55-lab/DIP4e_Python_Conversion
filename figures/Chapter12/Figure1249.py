"""Figure 12.49 - Harris corner detection on building image."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from General.corner import corner
from libDIPUM.data_path import dip_data


def im2double(arr: np.ndarray) -> np.ndarray:
    """im2double."""
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


print("Running Figure1249...")

# Data
img_path = dip_data("building-600by600.tif")
IB = im2double(plt.imread(img_path))

# Harris corners
C1 = corner(IB)
C2 = corner(IB, SensitivityFactor=0.249)
C3 = corner(IB, SensitivityFactor=0.17, QualityLevel=0.05)
C4 = corner(IB, QualityLevel=0.05)
C5 = corner(IB, QualityLevel=0.07)

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for ax in axes:
    ax.imshow(IB, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

axes[1].plot(C1[:, 0] - 1, C1[:, 1] - 1, "yo", markersize=4)
axes[2].plot(C2[:, 0] - 1, C2[:, 1] - 1, "yo", markersize=4)
axes[3].plot(C3[:, 0] - 1, C3[:, 1] - 1, "yo", markersize=4)
axes[4].plot(C4[:, 0] - 1, C4[:, 1] - 1, "yo", markersize=4)
axes[5].plot(C5[:, 0] - 1, C5[:, 1] - 1, "yo", markersize=4)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1249.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()
