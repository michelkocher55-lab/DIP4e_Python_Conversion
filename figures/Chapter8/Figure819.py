# Figure819.py

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

# Data
f = imread(dip_data("Fig0819(a).tif")).astype(np.uint8)

# Convert to Gray code
g = np.bitwise_xor(f, np.right_shift(f, 1))

# Get all bit planes (bit 0 to bit 7)
planef = [np.bitwise_and(np.right_shift(f, k), 1) for k in range(8)]
planeg = [np.bitwise_and(np.right_shift(g, k), 1) for k in range(8)]

# Figure 1
fig1 = plt.figure(1, figsize=(10, 5))

plt.subplot(2, 4, 1)
plt.imshow(planef[7], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 7")
plt.axis("off")
plt.subplot(2, 4, 2)
plt.imshow(planef[6], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 6")
plt.axis("off")
plt.subplot(2, 4, 3)
plt.imshow(planef[5], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 5")
plt.axis("off")
plt.subplot(2, 4, 4)
plt.imshow(planef[4], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 4")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(planeg[7], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 7")
plt.axis("off")
plt.subplot(2, 4, 6)
plt.imshow(planeg[6], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 6")
plt.axis("off")  # fixed
plt.subplot(2, 4, 7)
plt.imshow(planeg[5], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 5")
plt.axis("off")
plt.subplot(2, 4, 8)
plt.imshow(planeg[4], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 4")
plt.axis("off")

fig1.tight_layout()

# Figure 2
fig2 = plt.figure(2, figsize=(10, 5))

plt.subplot(2, 4, 1)
plt.imshow(planef[3], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 3")
plt.axis("off")
plt.subplot(2, 4, 2)
plt.imshow(planef[2], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 2")
plt.axis("off")
plt.subplot(2, 4, 3)
plt.imshow(planef[1], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 1")
plt.axis("off")
plt.subplot(2, 4, 4)
plt.imshow(planef[0], cmap="gray", vmin=0, vmax=1)
plt.title("Original, bit 0")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(planeg[3], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 3")
plt.axis("off")
plt.subplot(2, 4, 6)
plt.imshow(planeg[2], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 2")
plt.axis("off")
plt.subplot(2, 4, 7)
plt.imshow(planeg[1], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 1")
plt.axis("off")
plt.subplot(2, 4, 8)
plt.imshow(planeg[0], cmap="gray", vmin=0, vmax=1)
plt.title("Gray coded, bit 0")
plt.axis("off")

fig2.tight_layout()

# Save figures
fig1.savefig("Figure819.png", dpi=150, bbox_inches="tight")
fig2.savefig("Figure820.png", dpi=150, bbox_inches="tight")

plt.show()
