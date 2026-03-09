import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

print("Running Figure434 (Phase Manipulation)...")

# Image loading
img_path = dip_data("integrated-ckt-damaged.tif")

f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]

f = img_as_float(f)

# Process
# F = fft2(f);
F = np.fft.fft2(f)

# a = angle(F);
a = np.angle(F)

# Alter phase angle
# g1 = real (ifft2(abs(F).*exp(-i*a))); % Negative of the phase.
# Python complex 'j'.
g1 = np.real(np.fft.ifft2(np.abs(F) * np.exp(-1j * a)))

# g2 = real (ifft2(abs(F).*exp(i*(0.25)*a))); % Phase times a constant.
g2 = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * 0.25 * a)))

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].imshow(f, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")
axes[0].axis("off")

# g1 (Negative Phase)
axes[1].imshow(g1, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Phase Angle * -1")
axes[1].axis("off")

# g2 (Phase * 0.25)
axes[2].imshow(g2, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Phase Angle * 0.25")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure434.png")
print("Saved Figure434.png")
plt.show()
