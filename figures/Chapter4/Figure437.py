import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("building-600by600.tif")
f = imread(img_path)

# Fourier transform
F = np.fft.fft2(f)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f, cmap="gray")
axes[0].set_title("f")
axes[0].axis("off")

spec = np.fft.fftshift(np.log(1 + np.abs(F)))
spec = spec - spec.min()
if spec.max() > 0:
    spec = spec / spec.max()
axes[1].imshow(spec, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"DFT(f), Size = {F.shape}")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure437.png")
plt.show()
