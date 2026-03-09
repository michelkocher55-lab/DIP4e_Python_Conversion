import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.intScaling4e import intScaling4e
from libDIPUM.recnotch import recnotch
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("Saturnringe.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Fourier transform
F = np.fft.fft2(f)
S = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(F))))

# Filter design (centered)
H = np.fft.fftshift(recnotch("reject", "vertical", M, N, 5, 15))

# Filtering (no padding)
g = dftFiltering4e(f, H, padmode="none")

# Display (slightly gray)
H_disp = 0.98 * H

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].axis("off")

axes[0, 1].imshow(S, cmap="gray")
axes[0, 1].axis("off")

axes[1, 0].imshow(H_disp, cmap="gray")
axes[1, 0].axis("off")

axes[1, 1].imshow(g, cmap="gray")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure465.png")
plt.show()
