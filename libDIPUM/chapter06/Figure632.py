import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.wavefast import wavefast
from libDIPUM.wavecut import wavecut
from libDIPUM.waveback import waveback
from libDIPUM.wavedisplay import wavedisplay
from libDIPUM.data_path import dip_data

# Data
f = img_as_float(imread(dip_data("sinePulses.tif")))

# Wavelet transform
c, s = wavefast(f, 2, "sym4")

# Zeroing approximation coefficients
nc, y = wavecut("a", c, s)

# Back to image domain
f1 = waveback(nc, s, "sym4")

# Zeroing approximation and horizontal details coefficients
nc1 = c.copy()
Ix = np.where(nc == 0)[0]
nc1[: 2 * len(Ix)] = 0

# Back to image domain
f2 = waveback(nc1, s, "sym4")

# Display
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].axis("off")

axes[0, 1].imshow(wavedisplay(c, s, 3), cmap="gray")
axes[0, 1].axis("off")

axes[0, 2].imshow(wavedisplay(nc, s, 3), cmap="gray")
axes[0, 2].axis("off")

axes[1, 0].imshow(f1, cmap="gray")
axes[1, 0].axis("off")

axes[1, 1].imshow(wavedisplay(nc1, s, 3), cmap="gray")
axes[1, 1].axis("off")

axes[1, 2].imshow(f2, cmap="gray")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Figure632.png")
plt.show()
