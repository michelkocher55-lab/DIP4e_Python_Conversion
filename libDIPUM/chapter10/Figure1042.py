import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from helpers.multithresh import multithresh
from helpers.imquantize import imquantize
from libDIPUM.data_path import dip_data

# Data
image_path = dip_data("iceberg.tif")
f = imread(image_path)
if f.ndim == 3:
    f = f[:, :, 0]

# Histogram
hist, bins = np.histogram(f, bins=256, range=(0, 255))
h = hist / np.sum(hist)

print("Running multithresh(f, 2)...")
T = multithresh(f, 2)
print(f"Thresholds: {T}")

g = imquantize(f, T)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# 1. Original
axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

# 2. Histogram
# hist(double(f(:)), 256)
axes[1].plot(hist)
axes[1].set_title("Histogram")
axes[1].set_aspect("auto")
axes[1].set_xlim([0, 255])

# 3. Quantized
# imshow(g, [])
# g has values 1, 2, 3.
# auto-scaling with imshow(..., []) handled by matplotlib default or vmin/vmax
axes[2].imshow(g, cmap="gray")
axes[2].set_title("Multilevel Thresholding")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure1042.png")
print("Saved Figure1042.png")
plt.show()
