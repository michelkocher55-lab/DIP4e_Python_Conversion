import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve
import ia870 as ia
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("turbineblad-with-blk-dot.tif")

f = img_as_float(imread(img_path))

# Laplacian kernel
# w = [-1 -1 -1;-1 8 -1;-1 -1 -1];
w = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)

# Filtering
# h = imfilter(f,w,'replicate');
# scipy.ndimage.convolve uses 'nearest' for replicate padding typically?
# mode='nearest' replicates the edge value.
h = convolve(f, w, mode="nearest")

ha = np.abs(h)

# Thresholding
# T = 0.9*max(ha(:));
# g = ha >= T;
T = 0.9 * ha.max()
g = ha >= T

# Display
# subplot (1, 3, 1); imshow(f);
# subplot (1, 3, 2); imshow(h);
# subplot (1, 3, 3); imshow(morphoDilate4e(g, ones (7)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("f")
axes[0].axis("off")

# h might have negative values, standard imshow scales min-max
axes[1].imshow(h, cmap="gray")
axes[1].set_title("Laplacian Filtered (h)")
axes[1].axis("off")

# Dilate point for visibility
B = ia.iasebox(3)
g_dilated = ia.iadil(g, B)

axes[2].imshow(g_dilated, cmap="gray")
axes[2].set_title("Thresholded & Dilated Point")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure104.png")
print("Saved Figure104.png")
plt.show()
