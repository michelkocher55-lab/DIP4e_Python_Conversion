import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.ishade import ishade
from libDIPUM.data_path import dip_data

print("Running Figure1034 (Shading with ramp)...")

# Data
img_path = dip_data("Fig1036(b)(gaussian_noise_mean_0_std_10_added).tif")
fn_orig = imread(img_path)
if fn_orig.ndim == 3:
    fn_orig = fn_orig[:, :, 0]

M, N = fn_orig.shape

# r = ishade(M, N, 0.2, 0.6,'ramp', 0);
r = ishade(M, N, 0.2, 0.6, "ramp", 0)

# fs=immultiply(im2double(fn),r);
fn = img_as_float(fn_orig)
fs = fn * r

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(fn, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(r, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Shading Ramp")
axes[1].axis("off")

axes[2].imshow(fs, cmap="gray")
axes[2].set_title("Shaded Image")
axes[2].axis("off")

# Histograms
# Range [0, 1]

axes[3].hist(fn.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
axes[3].set_title("Histogram (Original)")
axes[3].set_xlim([0, 1])

axes[4].hist(r.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
axes[4].set_title("Histogram (Ramp)")
axes[4].set_xlim([0, 1])

axes[5].hist(fs.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
axes[5].set_title("Histogram (Shaded)")
axes[5].set_xlim([0, 1])

plt.tight_layout()
plt.savefig("Figure1034.png")
print("Saved Figure1034.png")
plt.show()
