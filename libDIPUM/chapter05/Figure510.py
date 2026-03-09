import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from libDIPUM.imnoise2 import imnoise2
from libDIPUM.spfilt import spfilt
from libDIPUM.data_path import dip_data

# Parameters
kernel_size = 3
p_salt = 0.05
p_pepper = 0.05

# Data
img_path = dip_data("circuitboard.tif")
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

# Noise adding
fnSaltPepper, _ = imnoise2(f, "salt & pepper", p_salt, p_pepper)

# Iterative median filtering (matches MATLAB: each iteration filters the same noisy image)
fHat = []
for _ in range(3):
    fHat.append(spfilt(fnSaltPepper, "median", kernel_size, kernel_size))

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

axes[0, 0].imshow(fnSaltPepper, **imshow_kwargs)
axes[0, 0].set_title("Original image")
axes[0, 0].axis("off")

titles = ["median filter 1 pass", "median filter 2 passes", "median filter 3 passes"]
for idx in range(3):
    ax = axes.flat[idx + 1]
    ax.imshow(fHat[idx], **imshow_kwargs)
    ax.set_title(titles[idx])
    ax.axis("off")

plt.tight_layout()
plt.savefig("Figure510.png")
plt.show()
