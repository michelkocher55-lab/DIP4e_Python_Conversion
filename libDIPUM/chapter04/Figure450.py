import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.lpfilter import lpfilter
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("satellite_original.tif")
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Filter design
H1 = np.fft.ifftshift(lpfilter("gaussian", P, Q, 50))
H2 = np.fft.ifftshift(lpfilter("gaussian", P, Q, 20))

# Filtering
g50 = dftFiltering4e(f, H1)
g20 = dftFiltering4e(f, H2)

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(f, cmap="gray")
axes[0].axis("off")

axes[1].imshow(g50, cmap="gray")
axes[1].axis("off")

axes[2].imshow(g20, cmap="gray")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure450.png")
plt.show()
