import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from libDIPUM.imnoise2 import imnoise2
from libDIPUM.spfilt import spfilt
from libDIPUM.data_path import dip_data

# Parameters
kernel_size = 3
mean = 0
std = 0.1

# Data
img_path = dip_data("circuitboard.tif")
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

# Noise adding
fn, _ = imnoise2(f, "gaussian", mean, std)

# Filtering
fHatArithMean = spfilt(fn, "amean", kernel_size, kernel_size)
fHatGeoMean = spfilt(fn, "gmean", kernel_size, kernel_size)

# Display
# show_image_window(f, "Original")
# show_image_window(fn, "Noisy")
# show_image_window(fHatArithMean, "Arithmetic Mean Filter")
# show_image_window(fHatGeoMean, "Geometric Mean Filter")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[0, 0].axis("off")
axes[0, 0].set_title("Original")

axes[0, 1].imshow(fn, cmap="gray", interpolation="nearest", resample=False)
axes[0, 1].axis("off")
axes[0, 1].set_title("Gaussian noise")

axes[1, 0].imshow(fHatArithMean, cmap="gray")
axes[1, 0].axis("off")
axes[1, 0].set_title("Arithmetic mean")

axes[1, 1].imshow(fHatGeoMean, cmap="gray")
axes[1, 1].axis("off")
axes[1, 1].set_title("Geometric mean")

plt.tight_layout()
plt.savefig("Figure57.png")
plt.show()
