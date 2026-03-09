import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from libDIPUM.cnotch import cnotch
from libDIPUM.imnoise3 import imnoise3
from libDIP.intScaling4e import intScaling4e
from libDIPUM.dftfilt import dftfilt
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("astronaut.tif")

f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

M, N = f.shape

# Generate sinusoidal noise pattern (r), its DFT (R), and its spectrum (S)
r, R, S = imnoise3(M, N, [25, 25], A=[0.3])
g = f + r
gs = intScaling4e(g)

# Compute spectrum of g
G = np.fft.fftshift(np.abs(np.fft.fft2(g)))
# Follow MATLAB: 1 + log(G). Use small epsilon to avoid log(0) warnings.
Glog = intScaling4e(1 + np.log(G + 1e-9))

# Create notch filters
# MATLAB: [M/2+1+25, N/2+1+25] (1-based). For Python 0-based, use M//2 + 25, N//2 + 25.
impulse_loc = [M // 2 + 25, N // 2 + 25]
H = cnotch("ideal", "reject", M, N, [impulse_loc], 2)
Hc = intScaling4e(np.fft.fftshift(H))

# Filter image
gf = intScaling4e(dftfilt(g, H))

# Display results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(gs, cmap="gray")
axes[0, 0].axis("off")

axes[0, 1].imshow(Glog, cmap="gray")
axes[0, 1].axis("off")

axes[1, 0].imshow(Hc, cmap="gray")
axes[1, 0].axis("off")

axes[1, 1].imshow(gf, cmap="gray")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure55.png")
plt.show()
