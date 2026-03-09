import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.intScaling4e import intScaling4e
from libDIPUM.recnotch import recnotch
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("cassini-interference.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Fourier transform
F = np.fft.fft2(f)

# Filter design (uncentered)
Hpass = recnotch("pass", "vertical", M, N, 5, 15)

# Apply filter (uncentered)
P = Hpass * F
pattern = intScaling4e(np.real(np.fft.fftshift(np.fft.ifft2(P))))

SP = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(P))))

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(SP, cmap="gray")
axes[0].axis("off")

axes[1].imshow(pattern, cmap="gray")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure466.png")
plt.show()
