import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIP.imRecon4e import imRecon4e
from libDIPUM.data_path import dip_data

# Parameters
# Angles32 = 5.625 * (0 : 31);
Angles32 = 5.625 * np.arange(32)
# Angles64 = 2.8125 * (0 : 63);
Angles64 = 2.8125 * np.arange(64)

# Data
img_name = dip_data("ellipse_and_circle.tif")
f = imread(img_name)

# Reconstructions
print("Computing BackProj (90)...")
rec90 = imRecon4e(f, 90)

print("Computing BackProj (0, 90)...")
rec0_90 = imRecon4e(f, [0, 90])

print("Computing BackProj (0, 45, 90, 135)...")
rec0_45_90_135 = imRecon4e(f, [0, 45, 90, 135])

print("Computing BackProj (32 angles)...")
# imRecon4e might be slow for many angles, but it implements waitbar roughly by printing (or not).
rec32 = imRecon4e(f, Angles32)

print("Computing BackProj (64 angles)...")
rec64 = imRecon4e(f, Angles64)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("f")
axes[0, 0].axis("off")

axes[0, 1].imshow(rec90, cmap="gray")
axes[0, 1].set_title("BackProj (90)")
axes[0, 1].axis("off")

axes[0, 2].imshow(rec0_90, cmap="gray")
axes[0, 2].set_title("BackProj (0, 90)")
axes[0, 2].axis("off")

axes[1, 0].imshow(rec0_45_90_135, cmap="gray")
axes[1, 0].set_title("BackProj (0, 45, 90, 135)")
axes[1, 0].axis("off")

axes[1, 1].imshow(rec32, cmap="gray")
axes[1, 1].set_title("BackProj (32 angles)")
axes[1, 1].axis("off")

axes[1, 2].imshow(rec64, cmap="gray")
axes[1, 2].set_title("BackProj (64 angles)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Figure534.png")
print("Saved Figure534.png")
plt.show()
