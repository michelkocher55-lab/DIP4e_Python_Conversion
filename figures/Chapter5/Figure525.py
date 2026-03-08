import os
import sys
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from libDIP.dftFiltering4e import dftFiltering4e
from General.atmosphturb import atmosphturb
from libDIPUM.data_path import dip_data

# Parameters
k_vals = [0.0025, 0.001, 0.00025]

# Data
img_path = dip_data('aerial_view_no_turb.tif')
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

M, N = f.shape

# Atmospheric perturbations
H = []
for k in k_vals:
    H.append(atmosphturb(2 * M, 2 * N, k))

# Filtering
g = []
for idx in range(len(k_vals)):
    g.append(dftFiltering4e(f, H[idx]))

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
imshow_kwargs = dict(cmap='gray', vmin=0, vmax=1, interpolation='nearest')

axes[0, 0].imshow(f, **imshow_kwargs)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, k in enumerate(k_vals):
    ax = axes.flat[i + 1]
    ax.imshow(g[i], **imshow_kwargs)
    ax.set_title(f"k = {k}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('Figure525.png')
plt.show()