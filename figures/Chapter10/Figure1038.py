import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
from scipy.ndimage import uniform_filter
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("Fig1041(a)(septagon_small_noisy_mean_0_stdv_10).tif")
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = f_orig[:, :, 0]

f = img_as_float(f_orig)

# Thresholding 1
T1 = threshold_otsu(f)
g1 = f > T1
print(f"Otsu Threshold 1: {T1}")

# Smooth
fs = uniform_filter(f, size=5, mode="nearest")

# Thresholding 2
T2 = threshold_otsu(fs)
g2 = fs > T2
print(f"Otsu Threshold 2 (Smoothed): {T2}")

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Noisy Image")
axes[0].axis("off")

axes[1].hist(f.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
axes[1].set_title("Histogram (Original)")
axes[1].set_xlim([0, 1])

axes[2].imshow(g1, cmap="gray")
axes[2].set_title(f"Otsu Thresholded (T={T1:.3f})")
axes[2].axis("off")

axes[3].imshow(fs, cmap="gray")
axes[3].set_title("Smoothed Image (5x5)")
axes[3].axis("off")

axes[4].hist(fs.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
axes[4].set_title("Histogram (Smoothed)")
axes[4].set_xlim([0, 1])

axes[5].imshow(g2, cmap="gray")
axes[5].set_title(f"Otsu Thresholded Smoothed (T={T2:.3f})")
axes[5].axis("off")

plt.tight_layout()
plt.savefig("Figure1038.png")
print("Saved Figure1038.png")
plt.show()
