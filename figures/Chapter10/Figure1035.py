import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

print("Running Figure1035 (Iterative Global Thresholding)...")

# Data
img_path = dip_data("fingerprint.tif")
f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]

# Convert to float for mean calculations
f = f.astype(float)

# Initial Threshold
T = np.mean(f)
print(f"Initial T: {T}")

done = False
count = 0
while not done:
    count += 1
    g = f > T

    # Mean of foreground (g) and background (~g)
    # Handle case where one might be empty (though unlikely for T=mean)
    if np.any(g):
        mean_fg = np.mean(f[g])
    else:
        mean_fg = 0  # Should not happen usually

    if np.any(~g):
        mean_bg = np.mean(f[~g])
    else:
        mean_bg = 0

    T_next = 0.5 * (mean_fg + mean_bg)

    done = abs(T - T_next) < 0.5
    T = T_next

print(f"Converged T: {T} after {count} iterations.")

# Final segmentation
g = f > T

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

# Histogram
axes[1].hist(f.ravel(), bins=256, color="k", histtype="step")
axes[1].axvline(x=T, color="r", linestyle="--", label=f"T={T:.1f}")
axes[1].set_title("Histogram")
axes[1].legend()

axes[2].imshow(g, cmap="gray")
axes[2].set_title(f"Global Thresholding (T={T:.1f})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure1035.png")
print("Saved Figure1035.png")
plt.show()
