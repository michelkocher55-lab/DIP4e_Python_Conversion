
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte
from libDIPUM.exacthist import exacthist
from libDIPUM.fun2hist import fun2hist
from libDIPUM.data_path import dip_data

def imhist(img):
    """Compute histogram for uint8 image with 256 bins [0, 255]."""
    # Create 256 bins
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    return hist

# Data Loading
img_path = dip_data('mars_moon_phobos.tif')
f = imread(img_path)
if f.ndim == 3: f = f[:,:,0]

# Ensure f is uint8? exacthist expects uint8 usually, or converts internally?
# exacthist requires integer values usually (0-255).
# If file is uint8, fine.
# MATLAB `imhist` works on uint8.

if f.dtype != np.uint8:
    # Convert to uint8 only if range allows, but exact specs might work on other ranges.
    # However, typically exacthist for images assumes discrete levels.
    # Let's assume input is uint8 or convertible.
    f = img_as_ubyte(f)

M, N = f.shape

# Histogram of f
Hf = imhist(f)

# Histogram Specification
#H(1:256) = M*N/256
H_target = np.full(256, (M * N) / 256.0)

#H = fun2hist(H, M*N)
# fun2hist standardizes the histogram to sum exactly to MN and be integers.
H_target = fun2hist(H_target, M * N)

# g = exacthist(f, H)
g = exacthist(f, H_target)
g = g[0]

# Hz = imhist(g)
Hg = imhist(g)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Plot histogram. MATLAB uses bar or stem or plot? "plot(Hf)"
axes[0, 1].plot(Hf, color='black')
axes[0, 1].set_title('Histogram of f')
axes[0, 1].set_xlim([0, 255])

axes[1, 0].plot(Hg, color='black')
axes[1, 0].set_title('Histogram of g')
axes[1, 0].set_xlim([0, 255])

axes[1, 1].imshow(g, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Exact Hist. Spec. Result')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure330.png')
print("Saved Figure330.png")
plt.show()