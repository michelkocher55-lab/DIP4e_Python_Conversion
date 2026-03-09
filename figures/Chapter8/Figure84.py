import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image

from libDIPUM.compare import compare
from libDIPUM.data_path import dip_data

# Figure84

# Parameters
Quality = [74, 10, 14]
Name = ["Figure84a.jpg", "Figure84b.jpg", "Figure84c.jpg"]

# Data
img_path = dip_data("Fig0801(a).tif")
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

f = imread(img_path)

# Compression, decompression
fHat = []
RMSE = []
for q, name in zip(Quality, Name):
    Image.fromarray(f).save(name, format="JPEG", quality=q)
    rec = imread(name)
    fHat.append(rec)
    RMSE.append(compare(f, rec, 0))

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(len(Quality)):
    fi = np.asarray(fHat[i])
    axes[i].imshow(fi, cmap="gray", vmin=fi.min(), vmax=fi.max())
    axes[i].set_title(f"Q = {Quality[i]}, RMSE = {RMSE[i]:.4f}")
    axes[i].axis("off")

plt.tight_layout()

# Print to file
fig.savefig("Figure84.png", dpi=300, bbox_inches="tight")

plt.show()
