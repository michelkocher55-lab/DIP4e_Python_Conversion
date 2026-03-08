
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import exposure

base_dir = '/Users/michelkocher/michel/Data/DIP-DIPUM/DIP'
# Filenames key map matches Figure316
files_map = [
    ('dark', 'Pollen-dark.tif', 'Dark'),
    ('light', 'Pollen-light.tif', 'Light'),
    ('lowcontrast', 'pollen-lowcontrast.tif', 'Low Contrast'),
    ('highcontrast', 'Pollen-high-contrast.tif', 'High Contrast')
]

# Data storage
originals = []
equalized = []

for key, fname, title in files_map:
    path = os.path.join(base_dir, fname)
    img = imread(path)
    if img.ndim == 3: img = img[:,:,0]

    originals.append(img)

    # Histeq
    # exposure.equalize_hist returns float [0,1].
    # We convert back to uint8 [0,255] for consistency with MATLAB display.
    eq_img_float = exposure.equalize_hist(img)
    eq_img = (eq_img_float * 255).astype(np.uint8)

    equalized.append(eq_img)

# Display
# 3x4 grid.
# Row 1: Originals
# Row 2: Equalized
# Row 3: Histograms of Equalized

titles = [x[2] for x in files_map]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i in range(4):
    # Row 1: Original
    ax1 = axes[0, i]
    ax1.imshow(originals[i], cmap='gray', vmin=0, vmax=255)
    ax1.set_title(titles[i])
    ax1.axis('off')

    # Row 2: Equalized
    ax2 = axes[1, i]
    ax2.imshow(equalized[i], cmap='gray', vmin=0, vmax=255)
    # ax2.set_title('Equalized')
    ax2.axis('off')

    # Row 3: Histogram of Equalized
    ax3 = axes[2, i]
    img_eq = equalized[i]
    counts, bins = np.histogram(img_eq.ravel(), bins=256, range=(0, 255))

    ax3.bar(bins[:-1], counts, width=1, color='black', align='edge')
    ax3.set_xlim([0, 255])
    ax3.set_ylim([0, counts.max() * 1.05])
    ax3.set_box_aspect(1)

plt.tight_layout()
plt.savefig('Figure320.png')
print("Saved Figure320.png")
plt.show()