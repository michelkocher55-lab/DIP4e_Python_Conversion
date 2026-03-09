import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.exposure import equalize_hist
from libDIPUM.data_path import dip_data

# %% Fig. 3.25
# Histogram equalization of hidden horse image

# %% Data
f = np.array(Image.open(dip_data("hidden-horse.tif")))
if f.ndim == 3:
    f = f[..., 0]

# %% Obtain the histogram equalization intensity transformation function
hfn, _ = np.histogram(f.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
hfn = hfn.astype(float) / f.size
thf = np.cumsum(hfn)

# %% Histogram equalized image (256 levels)
g_float = equalize_hist(f, nbins=256)
g = np.clip(np.round(255 * g_float), 0, 255).astype(np.uint8)

# %% Normalized histogram of equalized image
hng, _ = np.histogram(g.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
hng = hng.astype(float) / g.size

# %% Display
fig = plt.figure(1, figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(255 * thf)
plt.axis("square")
plt.axis("tight")

plt.subplot(1, 3, 2)
plt.imshow(g, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.bar(np.arange(256), hng)
plt.axis("square")
plt.axis("tight")

plt.tight_layout()
fig.savefig("Figure325.png", dpi=150, bbox_inches="tight")
plt.show()
