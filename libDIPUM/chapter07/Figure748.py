import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIPUM.imnoise2 import imnoise2
from libDIP.rgb2hsi4e import rgb2hsi4e
from libDIPUM.data_path import dip_data

# %% Parameters
Ps = 0.05
Pp = 0.05

# %% Data
RGB = img_as_float(imread(dip_data("lenna-RGB.tif")))
R = RGB[:, :, 0]
G = RGB[:, :, 1]
B = RGB[:, :, 2]

# %% Noise adding
Gn, n = imnoise2(G, "salt & pepper", Ps, Pp)
_ = n
RGBn = np.stack((R, Gn, B), axis=2)

# %% Convert to HSI
HSI = rgb2hsi4e(RGBn)
Hn = HSI[:, :, 0]
Sn = HSI[:, :, 1]
In = HSI[:, :, 2]

# %% Display
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(RGBn)
plt.axis("off")
plt.title("Noisy RGB")

plt.subplot(2, 2, 2)
plt.imshow(Hn, cmap="gray", vmin=np.min(Hn), vmax=np.max(Hn))
plt.axis("off")
plt.title("Hue")

plt.subplot(2, 2, 3)
plt.imshow(Sn, cmap="gray", vmin=np.min(Sn), vmax=np.max(Sn))
plt.axis("off")
plt.title("Saturation")

plt.subplot(2, 2, 4)
plt.imshow(In, cmap="gray", vmin=np.min(In), vmax=np.max(In))
plt.axis("off")
plt.title("Intensity")

plt.tight_layout()
plt.savefig("Figure748.png")
plt.show()
