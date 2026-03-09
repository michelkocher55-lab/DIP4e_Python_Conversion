import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.imnoise2 import imnoise2
from libDIPUM.spfilt import spfilt
from libDIPUM.adpmedian import adpmedian
from libDIPUM.data_path import dip_data

# Parameters
KernelSize = 5
PSalt = 0.25
PPepper = 0.25
SMax = 7

# Data
img_path = dip_data("circuitboard.tif")
f = plt.imread(img_path)
# Convert RGB → grayscale if needed
if f.ndim == 3:
    f = f[..., 0]
# im2double equivalent
f = f.astype(np.float64)
if f.max() > 1.0:
    f /= 255.0

# Add noise
fn, R = imnoise2(f, "salt & pepper", PSalt, PPepper)

# Filtering
fHatMedian = spfilt(fn, "median", KernelSize, KernelSize)
fHatAdaptiveMedian = adpmedian(fn, SMax)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(fn, cmap="gray")
plt.title("Salt & Pepper noise")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(fHatMedian, cmap="gray")
plt.title("Median")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(fHatAdaptiveMedian, cmap="gray")
plt.title("Adaptive median")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(fn, cmap="gray")
plt.title("Salt & Pepper noise")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(fHatMedian, cmap="gray")
plt.title("Median")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(fHatAdaptiveMedian, cmap="gray")
plt.title("Adaptive median")
plt.axis("off")

plt.tight_layout()
plt.savefig("Figure514.png", dpi=300, bbox_inches="tight")
plt.close()
