import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import uniform_filter
from libDIPUM.data_path import dip_data

# Image loading
img_path = dip_data("characterTestPattern688.tif")
f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]

f_float = img_as_float(f)

# Box filters
# MATLAB imfilter default boundary is 0.
# scipy.ndimage.uniform_filter default mode is 'reflect'.
# We should set mode='constant', cval=0.0 to match MATLAB default 'imfilter'.

gbox3 = uniform_filter(f_float, size=3, mode="constant", cval=0.0)
gbox11 = uniform_filter(f_float, size=11, mode="constant", cval=0.0)
gbox21 = uniform_filter(f_float, size=21, mode="constant", cval=0.0)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")  # Original
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(gbox3, cmap="gray")
axes[1].set_title("Box 3x3")
axes[1].axis("off")

axes[2].imshow(gbox11, cmap="gray")
axes[2].set_title("Box 11x11")
axes[2].axis("off")

axes[3].imshow(gbox21, cmap="gray")
axes[3].set_title("Box 21x21")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("Figure339.png")
print("Saved Figure339.png")
plt.show()
