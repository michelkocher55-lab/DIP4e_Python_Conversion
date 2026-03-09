import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.motionBlurTF4e import motionBlurTF4e
from libDIPUM.data_path import dip_data

img_name = dip_data("original_DIP.tif")
f_orig = imread(img_name)
if f_orig.ndim == 3:
    f_orig = f_orig[:, :, 0]
f = img_as_float(f_orig)

M, N = f.shape

# Generate blurring filter
# H = fftshift (motionBlurTF4e (M, N, 0.1, 0.1, 1));
# motionBlurTF4e returns centered H.
# fftshift creates standard uncentered FFT format (DC at corner).
H_centered = motionBlurTF4e(M, N, 0.1, 0.1, 1)
H = np.fft.fftshift(H_centered)

# Get FFT of image
F = np.fft.fft2(f)

# Multiply
G = F * H

# Output
# g = real(ifft2(G));
g = np.real(np.fft.ifft2(G))

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(g, cmap="gray")
axes[0, 1].set_title("Blurred Image")
axes[0, 1].axis("off")

axes[1, 0].imshow(np.abs(H), cmap="gray")
axes[1, 0].set_title("Filter Magnitude |H|")
axes[1, 0].axis("off")

axes[1, 1].imshow(np.angle(H), cmap="gray")
axes[1, 1].set_title("Filter Phase angle(H)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure526.png")
print("Saved Figure526.png")
plt.show()
