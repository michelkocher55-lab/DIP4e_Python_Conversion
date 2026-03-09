import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

from libDIP.im2jpeg4e import im2jpeg4e
from libDIP.jpeg2im4e import jpeg2im4e
from libDIPUM.imratio import imratio
from libDIPUM.compare import compare
from libDIPUM.data_path import dip_data

# %% Parameters
Quality = 20

# %% Data
RGB = imread(dip_data("Fig0604(a)(iris).tif"))
R = RGB[:, :, 0]
G = RGB[:, :, 1]
B = RGB[:, :, 2]

# %% Compression JPEG
y = im2jpeg4e(R, Quality)
RHat = jpeg2im4e(y)
CR_R = imratio(R, y)
RMSE_R = compare(R, RHat, 0)

y = im2jpeg4e(G, Quality)
GHat = jpeg2im4e(y)
CR_G = imratio(G, y)
RMSE_G = compare(G, GHat, 0)

y = im2jpeg4e(B, Quality)
BHat = jpeg2im4e(y)
CR_B = imratio(B, y)
RMSE_B = compare(B, BHat, 0)

RGBCompressed = np.stack((RHat, GHat, BHat), axis=2)
_ = (RMSE_R, RMSE_G, RMSE_B)

# %% Display
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(RGB)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(RGBCompressed)
plt.axis("off")
plt.title(f"CR = {CR_R:.3g}, {CR_G:.3g}, {CR_B:.3g}")

plt.tight_layout()
plt.savefig("Figure749.png")
plt.show()
