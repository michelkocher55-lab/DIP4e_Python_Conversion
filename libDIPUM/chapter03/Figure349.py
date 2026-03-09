import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import correlate
from scipy.signal import medfilt2d

from libDIPUM.imnoise2 import imnoise2
from libDIPUM.gaussiankernel import gaussiankernel
from libDIPUM.data_path import dip_data

# Data
f = imread(dip_data("circuitboard.tif"))
if f.ndim == 3:
    f = f[:, :, 0]

# Add noise
fn, _ = imnoise2(f, "salt & pepper")

# Linear filtering
w, _ = gaussiankernel(7, "sampled", 3.0, 1.0)
w = w / np.sum(w)
gG = correlate(fn, w, mode="reflect")  # MATLAB 'symmetric'

# Non linear filtering
gM = medfilt2d(fn, kernel_size=7)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(fn, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gG, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gM, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.tight_layout()
plt.savefig("Figure349.png")
print("Saved Figure349.png")
plt.show()
