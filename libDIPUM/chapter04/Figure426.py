import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("boy.tif")
f_orig = imread(img_path)

# MATLAB imresize preserves range for double; use preserve_range and then convert to float
f_resized = resize(
    f_orig,
    (1774, 1546),
    order=1,
    mode="reflect",
    anti_aliasing=True,
    preserve_range=True,
)
f = img_as_float(f_resized)

# Rectangle
g = np.zeros((1774, 1546), dtype=float)
# MATLAB 722:1052, 749:797 -> Python 0-based
g[721:1052, 748:797] = 1

# Fourier transform
F = np.fft.fft2(f)
G = np.fft.fft2(g)

# Reconstruct using mixed magnitude/phase
RecGModulusFAngle = np.real(np.fft.ifft2(np.abs(G) * np.exp(1j * np.angle(F))))
RecFModulusGAngle = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * np.angle(G))))

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("f")
axes[0, 0].axis("off")

axes[0, 1].imshow(RecFModulusGAngle, cmap="gray")
axes[0, 1].set_title("Rec by using |F| and arg (G)")
axes[0, 1].axis("off")

axes[1, 0].imshow(g, cmap="gray")
axes[1, 0].set_title("g")
axes[1, 0].axis("off")

axes[1, 1].imshow(RecGModulusFAngle, cmap="gray")
axes[1, 1].set_title("Rec by using |G| and arg (F)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure426.png")
plt.show()
