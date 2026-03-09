import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from General.kmeans import kmeans
from libDIPUM.data_path import dip_data

# Data
image_path = dip_data("book-cover.tif")
f_raw = imread(image_path)
if f_raw.ndim == 3:
    pass
f = img_as_float(f_raw)
f_flat = f.flatten()

print("Running K-means...")
idx, centers = kmeans(f_flat, 3)

# Reconstruct
idx_reshaped = idx.reshape(f.shape)
fseg = idx_reshaped.astype(np.float64)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

if f_raw.ndim == 2:
    axes[0].imshow(f, cmap="gray")
else:
    axes[0].imshow(f)
axes[0].set_title("Original Image")
axes[0].axis("off")

# MATLAB: imshow(fseg, [], 'InitialMagnification', 'fit')
# [] means scale display range to [min(fseg) max(fseg)]
axes[1].imshow(fseg, cmap="gray")
axes[1].set_title("Segmented Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure1049.png")
print("Saved Figure1049.png")
plt.show()
