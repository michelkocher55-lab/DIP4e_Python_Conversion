import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.levelSetForce4e import levelSetForce4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("rose479by512.tif")
f = img_as_float(imread(img_path))

# Compute edge-marking function.
W = levelSetForce4e("gradient", [f, 1, 50])

# Display
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(f, cmap="gray")
axes[0].axis("off")
axes[1].imshow(W, cmap="gray")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure1128.png")
plt.show()
