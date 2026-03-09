import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import histogram
from libDIPUM.exacthist import exacthist
from libDIPUM.fun2hist import fun2hist
from libDIPUM.trapezmf import trapezmf
from libDIPUM.data_path import dip_data

# Image loading
img_path = dip_data("hidden-horse.tif")
f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]

M, N = f.shape

# Generate function: uniform then ramp tail.
# z = 1:256
z = np.arange(1, 257)
# trapezmf(z, 0, 0, 64, 256)
# This means:
# a=0, b=0 (flat start)
# c=64 (start of ramp down)
# d=256 (end of ramp down)
# So flat 1.0 from 0 to 64, then ramp down to 0 at 256.
fun = trapezmf(z, 0, 0, 64, 256)

# Generate histogram
H = fun2hist(fun, M * N)

# Generate image with this histogram
# exacthist(f, H) returns (g, Hg, lexiOrder)
# Note: exacthist.m returns [g, Hg, lexiOrder]. My internal implementation of exacthist.py does same.
# MATLAB script calls: gramp = exacthist(...)
# If function returns multiple values, MATLAB assigns first to gramp.
# Python: tuple unpacking or index [0].

result = exacthist(f, H)
gramp = result[0]

# Calculate histogram of result for display check
counts, centers = histogram(gramp, nbins=256, source_range="image")

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Specified Histogram H
axes[0].bar(np.arange(256), H, width=1)
axes[0].set_title("Specified Histogram H")
axes[0].set_xlim([0, 255])

# 2. Result Image
axes[1].imshow(gramp, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Result gramp")
axes[1].axis("off")

# 3. Result Histogram Hg
axes[2].bar(centers, counts, width=1)
axes[2].set_title("Result Histogram Hg")
axes[2].set_xlim([0, 255])

plt.tight_layout()
plt.savefig("Figure329.png")
print("Saved Figure329.png")
plt.show()
