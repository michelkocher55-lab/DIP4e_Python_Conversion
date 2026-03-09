import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.lpfilter import lpfilter
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data

# Parameters
D0 = [10, 30, 60, 160, 460]

# Data
img_path = dip_data("characterTestPattern688.tif")
f = img_as_float(imread(img_path))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Process
results = []
for d0 in D0:
    H = np.fft.fftshift(lpfilter("gaussian", P, Q, d0))
    results.append(dftFiltering4e(f, H))

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for idx, d0 in enumerate(D0):
    ax = axes.flat[idx + 1]
    ax.imshow(results[idx], cmap="gray")
    ax.set_title(f"D0 = {d0}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("Figure444.png")
plt.show()
