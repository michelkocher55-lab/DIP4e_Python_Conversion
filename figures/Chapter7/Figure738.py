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

# %% Extract individual RGB and HSI components
r = f[:, :, 0]
g = f[:, :, 1]
b = f[:, :, 2]

# %% Transform to HSI
H = rgb2hsi4e(f)
h = H[:, :, 0]
s = H[:, :, 1]
i = H[:, :, 2]

# %% Filter individual RGB components
w = np.ones((5, 5), dtype=float) / 25.0
rf = convolve(r, w, mode="nearest")
gf = convolve(g, w, mode="nearest")
bf = convolve(b, w, mode="nearest")

# %% Convert back to RGB format
fRGB_filtered = np.stack((rf, gf, bf), axis=2)

# %% Filter Intensity component of HSI image
If = convolve(i, w, mode="nearest")

# %% Convert back to HSI
fHSI_filtered = np.stack((h, s, If), axis=2)

# %% Convert to RGB for comparisons
fHSI_filtered = hsi2rgb4e(fHSI_filtered)

# %% Convert to gray so that differences show clearly
f1 = (
    0.2989 * fRGB_filtered[:, :, 0]
    + 0.5870 * fRGB_filtered[:, :, 1]
    + 0.1140 * fRGB_filtered[:, :, 2]
)
f2 = (
    0.2989 * fHSI_filtered[:, :, 0]
    + 0.5870 * fHSI_filtered[:, :, 1]
    + 0.1140 * fHSI_filtered[:, :, 2]
)
d = f1 - f2

# %% Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(fRGB_filtered)
plt.axis("off")
plt.title("each component R, G and B are filtered")

plt.subplot(1, 3, 2)
plt.imshow(fHSI_filtered)
plt.axis("off")
plt.title("Only the intensity is filtered")

plt.subplot(1, 3, 3)
plt.imshow(d, cmap="gray", vmin=np.min(d), vmax=np.max(d))
plt.axis("off")
plt.title("Difference between the 2")

plt.tight_layout()
plt.savefig("Figure738.png")
plt.show()
