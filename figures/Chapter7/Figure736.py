import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

# %% Data
img_path = dip_data("lenna-RGB.tif")
f = img_as_float(imread(img_path))

# %% Convert to RGB
R = f[:, :, 0]
G = f[:, :, 1]
B = f[:, :, 2]

# %% Display
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(f)
plt.axis("off")
plt.title("RGB")

plt.subplot(2, 2, 2)
plt.imshow(R, cmap="gray")
plt.axis("off")
plt.title("R")

plt.subplot(2, 2, 3)
plt.imshow(G, cmap="gray")
plt.axis("off")
plt.title("G")

plt.subplot(2, 2, 4)
plt.imshow(B, cmap="gray")
plt.axis("off")
plt.title("B")

plt.tight_layout()
plt.savefig("Figure736.png")
plt.show()
