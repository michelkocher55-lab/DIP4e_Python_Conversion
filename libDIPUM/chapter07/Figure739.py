import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.io import imread
from skimage.util import img_as_float

from libDIP.rgb2hsi4e import rgb2hsi4e
from libDIP.hsi2rgb4e import hsi2rgb4e
from libDIPUM.data_path import dip_data

# %% Data
img_path = dip_data("lenna-RGB.tif")
f = img_as_float(imread(img_path))

# %% Convert to RGB
R = f[:, :, 0]
G = f[:, :, 1]
B = f[:, :, 2]

# %% Transform to HSI
H = rgb2hsi4e(f)
h = H[:, :, 0]
s = H[:, :, 1]
i = H[:, :, 2]

# %% Laplacian kernel
w = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)

# %% Filtering in RGB domain
LR = convolve(R, w, mode="nearest")
LG = convolve(G, w, mode="nearest")
LB = convolve(B, w, mode="nearest")
fRGB_filtered = np.stack((LR, LG, LB), axis=2)

# %% Filtering in HSI domain
If = convolve(i, w, mode="nearest")
Hd = img_as_float(h)
Sd = img_as_float(s)
fHSI_filtered = np.stack((Hd, Sd, If), axis=2)

# %% Back to RGB domain
RGB1 = hsi2rgb4e(fHSI_filtered)

# %% Convert to gray so that differences will show up clearly
f1 = (
    0.2989 * fRGB_filtered[:, :, 0]
    + 0.5870 * fRGB_filtered[:, :, 1]
    + 0.1140 * fRGB_filtered[:, :, 2]
)
f2 = 0.2989 * RGB1[:, :, 0] + 0.5870 * RGB1[:, :, 1] + 0.1140 * RGB1[:, :, 2]
d = f1 - f2

# %% Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(fRGB_filtered)
plt.axis("off")
plt.title("each component R, G and B are filtered")

plt.subplot(1, 3, 2)
plt.imshow(RGB1)
plt.axis("off")
plt.title("Only the intensity is filtered")

plt.subplot(1, 3, 3)
plt.imshow(d, cmap="gray", vmin=np.min(d), vmax=np.max(d))
plt.axis("off")
plt.title("Difference between the 2")

plt.tight_layout()
plt.savefig("Figure739.png")
plt.show()
