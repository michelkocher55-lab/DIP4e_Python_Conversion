import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

f = img_as_float(imread(dip_data("Fig1007(a)(wirebond_mask).tif")))

# Laplacian kernel
# w = ones(3); w(2,2) = -8;
w = np.ones((3, 3), dtype=float)
w[1, 1] = -8.0

# Filtering
# g = imfilter(f, w, 'replicate');
g = convolve(f, w, mode="nearest")

# Scaling
# gs = intScaling4e(g);
gs = intScaling4e(g)

# Thresholding
# ga = abs(g);
# gp = g > 0;
ga = np.abs(g)
gp = g > 0

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("f")
axes[0].axis("off")

axes[1].imshow(gs, cmap="gray")
axes[1].set_title("Scaled Laplacian (gs)")
axes[1].axis("off")

# ga is float, imshow scales auto.
axes[2].imshow(ga, cmap="gray")
axes[2].set_title("abs(Laplacian)")
axes[2].axis("off")

axes[3].imshow(gp, cmap="gray")
axes[3].set_title("g > 0")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("Figure105.png")
plt.show()
