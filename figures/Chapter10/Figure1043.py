import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.localthresh import localthresh
from General.multithresh import multithresh
from General.imquantize import imquantize
from General.stdfilt import stdfilt
from libDIPUM.data_path import dip_data

# Data
image_path = dip_data("yeast-cells.tif")
f = imread(image_path)
if f.ndim == 3:
    f = f[:, :, 0]

# Otsu multi thresholding
print("Running multithresh(f, 2)...")
T = multithresh(f, 2)
print(f"Thresholds: {T}")

g1 = imquantize(f, T)

# Local thresholding
nhood = np.ones((3, 3))
print("Running localthresh...")
g2 = localthresh(f, nhood, 30, 1.5, "global")
std = stdfilt(f, nhood)

# Display
fig, axes = plt.subplots(2, 2, figsize=(15, 6))

# 1. Original
axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

# 2. Otsu Multi (g1)
axes[0, 1].imshow(g1, cmap="gray")
axes[0, 1].set_title("Otsu Multi (g1)")
axes[0, 1].axis("off")

# 3. Local std
axes[1, 0].imshow(std, cmap="gray")
axes[1, 0].set_title("Otsu Multi (g1)")
axes[1, 0].axis("off")

# 4. Local Thresh (g2)
axes[1, 1].imshow(g2, cmap="gray")
axes[1, 1].set_title("Local Thresh (g2)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure1043.png")
print("Saved Figure1043.png")
plt.show()
