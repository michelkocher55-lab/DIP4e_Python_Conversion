import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIP.rgb2hsi4e import rgb2hsi4e
from libDIPUM.data_path import dip_data

# %% Data
img_path = dip_data("lenna-RGB.tif")
f = img_as_float(imread(img_path))

# %% Extract individual RGB and HSI components
r = f[:, :, 0]
g = f[:, :, 1]
b = f[:, :, 2]
_ = (r, g, b)

# %% Transform to HSI
H = rgb2hsi4e(f)
h = H[:, :, 0]
s = H[:, :, 1]
i = H[:, :, 2]

# %% Display
plt.figure(figsize=(10, 3.5))

plt.subplot(1, 3, 1)
plt.imshow(h, cmap="gray", vmin=np.min(h), vmax=np.max(h))
plt.axis("off")
plt.title("Hue")

plt.subplot(1, 3, 2)
plt.imshow(s, cmap="gray", vmin=np.min(s), vmax=np.max(s))
plt.axis("off")
plt.title("Saturation")

plt.subplot(1, 3, 3)
plt.imshow(i, cmap="gray", vmin=np.min(i), vmax=np.max(i))
plt.axis("off")
plt.title("Intensity")

plt.tight_layout()
plt.savefig("Figure737.png")
plt.show()
