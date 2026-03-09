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

# Noise adding (computed but not used in MATLAB display)
fnSaltPepper, _ = imnoise2(f, "salt & pepper", p_salt, p_pepper)

# Min max filtering
fMin = spfilt(f, "min", kernel_size, kernel_size)
fMax = spfilt(f, "max", kernel_size, kernel_size)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

axes[0].imshow(fMax, **imshow_kwargs)
axes[0].axis("off")
axes[0].set_title("Max filter")

axes[1].imshow(fMin, **imshow_kwargs)
axes[1].axis("off")
axes[1].set_title("Min filter")

plt.tight_layout()
plt.savefig("Figure511.png")
plt.show()
