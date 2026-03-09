from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_ubyte

from libDIPUM.exacthist import exacthist
from libDIPUM.fun2hist import fun2hist
from libDIPUM.sigmamf import sigmamf  # Assuming this exists based on ls
from libDIPUM.data_path import dip_data


def imhist(img: Any, bins: Any = 256):
    """imhist."""
    hist, _ = np.histogram(img.flatten(), bins, [0, bins])
    return hist


# Data Loading
img_path = dip_data("mars_moon_phobos.tif")
f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]
# Ensure uint8 for exacthist
if f.dtype != np.uint8:
    f = img_as_ubyte(f)

M, N = f.shape

# Mask
# idx = find(f == 0); mask = ones; mask(idx) = 0;
# Convert to boolean mask where True means "Process this pixel"
# exacthist mask: "Only pixels where mask > 0 are processed."
# So mask should be 0 for background.
mask = np.ones((M, N), dtype=bool)
mask[f == 0] = False

# Part 1: Exact Histogram Equalization (Uniform)
# H(1:256) = M*N/256 roughly.
# Note: mask reduces number of pixels?
# MATLAB: H(1:256) = M*N/256.
# Wait, if we use a mask, exacthist processes only masked pixels.
# The target H usually sums to the number of *masked* pixels or *total* pixels?
# exacthist documentation/implementation (which I wrote) checks sum against `num_active`.
# And normalizes H if mismatch.
# MATLAB script sets H based on M*N (total pixels).
# `fun2hist` (MATLAB) normalizes to M*N usually.
# If `exacthist` sees mask, it will likely see mismatch (Sum(H)=MN != NumMasked).
# My `exacthist.py` implementation handles this by rescaling H to match `num_active`.
# So strictly following MATLAB script is fine.

H_uniform = np.full(256, (M * N) / 256.0)
H_uniform = fun2hist(H_uniform, M * N)

# gmasked = exacthist(f, H, mask)
gmasked, _, _ = exacthist(f, H_uniform, mask)

# Part 2: Custom Histogram (Sigmoid)
# z = 1:256;
# sig = 0.065 + sigmamf(z, 32, 256);
# Check sigmamf signature in Python.
# Assuming sigmamf(x, a, c) or sigmamf(x, [a, c])?
# MATLAB: sigmamf(x, [a c]).
# MATLAB script: sigmamf(z, 32, 256). This implies 2 separate args?
# Or maybe custom `sigmamf`?
# I'll try calling with separate args `sigmamf(z, 32, 256)`. If it fails, I'll try list.
# Actually, let's peek lexiorder logic or similar? No time.
# I will assume standard python transcription of typical usage.

z = np.arange(1, 257)  # 1 to 256

# Try calling sigmamf.
try:
    # Assuming arguments a=32, c=256?
    # MATLAB sigmamf(x, [a c]).
    # But script says sigmamf(z,32,256).
    # Maybe the script uses a custom version that takes unpacked args.
    sig_vals = sigmamf(z, 32, 256)
except TypeError:
    # Retry with list
    sig_vals = sigmamf(z, [32, 256])

sig = 0.065 + sig_vals

# Hup = fun2hist(sig, M*N)
Hup = fun2hist(sig, M * N)

# [gup, Hg] = exacthist(f, Hup, mask)
gup, Hg, _ = exacthist(f, Hup, mask)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Mask
axes[0, 0].imshow(mask, cmap="gray")
axes[0, 0].set_title("Mask")
axes[0, 0].axis("off")

# 2. Masked Equalized
axes[0, 1].imshow(gmasked, cmap="gray")
axes[0, 1].set_title("Masked Hist. Eq.")
axes[0, 1].axis("off")

# 3. Hup Bar
axes[0, 2].bar(np.arange(256), Hup, color="black", width=1)
axes[0, 2].set_title("Target Histogram Hup")
axes[0, 2].set_xlim([0, 255])

# 4. Hg Bar (Result Hist)
axes[1, 0].bar(np.arange(256), Hg, color="black", width=1)
axes[1, 0].set_title("Result Histogram Hg")
axes[1, 0].set_xlim([0, 255])

# 5. gup (Result Image)
axes[1, 1].imshow(gup, cmap="gray")
axes[1, 1].set_title("Masked Specified Result (gup)")
axes[1, 1].axis("off")

# Empty slot
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Figure331.png")
print("Saved Figure331.png")
plt.show()
