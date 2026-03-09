import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIPUM.imnoise2 import imnoise2
from libDIP.rgb2hsi4e import rgb2hsi4e
from libDIPUM.data_path import dip_data

# %% Parameters
Mu = 0
Std = 28 / 255.0
Std = 20 / 255.0

# %% Data
RGB = img_as_float(imread(dip_data("lenna-RGB.tif")))
R = RGB[:, :, 0]
G = RGB[:, :, 1]
B = RGB[:, :, 2]

# %% Noise adding
Rn, n = imnoise2(R, "gaussian", Mu, Std)
Gn, n = imnoise2(G, "gaussian", Mu, Std)
Bn, n = imnoise2(B, "gaussian", Mu, Std)
_ = n

RGBn = np.stack((Rn, Gn, Bn), axis=2)

# %% Convert to HSI
HSI = rgb2hsi4e(RGBn)
Hn = HSI[:, :, 0]
Sn = HSI[:, :, 1]
In = HSI[:, :, 2]

# %% Display (Figure746)
fig1 = plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(Rn, cmap="gray")
plt.axis("off")
plt.title("Noisy R")

plt.subplot(2, 2, 2)
plt.imshow(Gn, cmap="gray", vmin=np.min(Gn), vmax=np.max(Gn))
plt.axis("off")
plt.title("Noisy G")

plt.subplot(2, 2, 3)
plt.imshow(Bn, cmap="gray", vmin=np.min(Bn), vmax=np.max(Bn))
plt.axis("off")
plt.title("Noisy B")

plt.subplot(2, 2, 4)
plt.imshow(RGBn)
plt.axis("off")
plt.title("Noisy RGB")

plt.tight_layout()
fig1.savefig("Figure746.png")

# %% Display (Figure747)
fig2 = plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(Hn, cmap="gray", vmin=np.min(Hn), vmax=np.max(Hn))
plt.axis("off")
plt.title("Hue")

plt.subplot(1, 3, 2)
plt.imshow(Sn, cmap="gray", vmin=np.min(Sn), vmax=np.max(Sn))
plt.axis("off")
plt.title("Saturation")

plt.subplot(1, 3, 3)
plt.imshow(In, cmap="gray", vmin=np.min(In), vmax=np.max(In))
plt.axis("off")
plt.title("Intensity")

plt.tight_layout()
fig2.savefig("Figure747.png")

plt.show()
