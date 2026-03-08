"""Figure 8.51 - Watermark attacks and correlation display."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.watermarkMKR import watermarkMKR
from libDIPUM.data_path import dip_data

print("Running Figure851 (watermark attack comparison)...")

# Parameters
WatermarkSize = 1000
HSize = 21
Sigma = 7
Alpha = 0.1

# Data (fixed path)
f_path = dip_data('lena.tif')
f = imread(f_path)
if f.ndim == 3:
    f = f[..., 0]
NR, NC = f.shape

# Watermark sequence
w = np.random.randn(WatermarkSize, 1)

# Add watermark with attacks
wi = {}
d = {}
c = {}

wi["JPEG70"], d["JPEG70"], c["JPEG70"] = watermarkMKR(f, w, Alpha, "jpeg70", "same")
wi["JPEG10"], d["JPEG10"], c["JPEG10"] = watermarkMKR(f, w, Alpha, "jpeg10", "same")
wi["Filter"], d["Filter"], c["Filter"] = watermarkMKR(f, w, Alpha, "filter", "same")
wi["noise"], d["noise"], c["noise"] = watermarkMKR(f, w, Alpha, "noise", "same")
wi["HistEq"], d["HistEq"], c["HistEq"] = watermarkMKR(f, w, Alpha, "heq", "same")
wi["Rotate"], d["Rotate"], c["Rotate"] = watermarkMKR(f, w, Alpha, "rotate", "same")

# Display
fig = plt.figure(1, figsize=(11, 7))

keys = ["JPEG70", "JPEG10", "Filter", "noise", "HistEq", "Rotate"]
for i, k in enumerate(keys, start=1):
    ax = fig.add_subplot(2, 3, i)
    ax.imshow(wi[k], cmap="gray")
    ax.set_title(f"c = {c[k]:.4f}")
    ax.axis("off")

# Save
out_path = os.path.join(os.path.dirname(__file__), "Figure851.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
