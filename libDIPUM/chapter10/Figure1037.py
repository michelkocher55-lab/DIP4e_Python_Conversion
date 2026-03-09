import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from libDIPUM.otsuthresh import otsuthresh
from libDIPUM.data_path import dip_data

print("Running Figure1037 (Otsu on Noisy Image w/ Smoothing)...")

img_path = dip_data("Fig1036(c)(gaussian_noise_mean_0_std_50_added).tif")
I = imread(img_path)
if I.ndim == 3:
    I = I[:, :, 0]

# 1. Noisy Image Stats
if I.dtype == np.uint8:
    h, bins = np.histogram(I, bins=256, range=(0, 255))
else:
    h, bins = np.histogram(I, bins=256)

T_norm, SM = otsuthresh(h)
if I.dtype == np.uint8:
    T = T_norm * 255
else:
    T = T_norm

print(f"Noisy Image: T={T:.2f}, SM={SM:.4f}")
g_noisy = I > T

# 2. Smoothing
I_smooth = gaussian_filter(I.astype(float), sigma=1.5)  # Sigma choice?

# Check normalized histogram of smoothed
h_s, bins_s = np.histogram(I_smooth, bins=256)
T_s_norm, SM_s = otsuthresh(h_s)

min_v, max_v = I_smooth.min(), I_smooth.max()
T_s = min_v + T_s_norm * (max_v - min_v)

print(f"Smoothed Image: T={T_s:.2f}, SM={SM_s:.4f}")
g_smooth = I_smooth > T_s

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Noisy
axes[0, 0].imshow(I, cmap="gray")
axes[0, 0].set_title("Noisy Image")
axes[0, 0].axis("off")

axes[0, 1].plot(np.arange(1, len(h) - 1), h[1:-1], "k")
axes[0, 1].axvline(x=T, color="r", linestyle="--")
axes[0, 1].set_title("Histogram (Noisy)")

axes[0, 2].imshow(g_noisy, cmap="gray")
axes[0, 2].set_title(f"Otsu (SM={SM:.2f})")
axes[0, 2].axis("off")

# Row 2: Smoothed
axes[1, 0].imshow(I_smooth, cmap="gray")
axes[1, 0].set_title("Smoothed Image (Gaussian)")
axes[1, 0].axis("off")

axes[1, 1].plot(h_s, "k")
axes[1, 1].axvline(x=T_s, color="r", linestyle="--")
axes[1, 1].set_title("Histogram (Smoothed)")

axes[1, 2].imshow(g_smooth, cmap="gray")
axes[1, 2].set_title(f"Otsu (SM={SM_s:.2f})")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Figure1037.png")
print("Saved Figure1037.png")
plt.show()
