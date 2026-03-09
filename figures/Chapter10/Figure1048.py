from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.splitmerge import splitmerge
from libDIPUM.data_path import dip_data


def predicate(region: Any):
    """predicate."""
    # sd = std2(region);
    # m = mean2(region);
    # flag = (sd > 10) & (m > 0) & (m < 125);
    # Note: MATLAB std2 is sample standard deviation (ddof=1).
    sd = np.std(region, ddof=1)
    m = np.mean(region)
    return (sd > 10) and (m > 0) and (m < 125)


# Data
image_path = dip_data("cygnusloop.tif")
f = imread(image_path)

# Split and Merge
print("Processing splitmerge(32)...")
g32 = splitmerge(f, 32, predicate)

print("Processing splitmerge(16)...")
g16 = splitmerge(f, 16, predicate)

print("Processing splitmerge(8)...")
g8 = splitmerge(f, 8, predicate)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(g32, cmap="gray")
axes[0, 1].set_title("g32")
axes[0, 1].axis("off")

axes[1, 0].imshow(g16, cmap="gray")
axes[1, 0].set_title("g16")
axes[1, 0].axis("off")

axes[1, 1].imshow(g8, cmap="gray")
axes[1, 1].set_title("g8")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure1048.png")
print("Saved Figure1048.png")
plt.show()
