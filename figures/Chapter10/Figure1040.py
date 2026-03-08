import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.gradlapthresh import gradlapthresh
from libDIPUM.otsuthresh import otsuthresh
from libDIPUM.data_path import dip_data

# Figure 10.40
# Threshold segmentation of yeast image using edge information.

# Data
f = imread(dip_data('yeast-cells.tif'))
if f.ndim == 3:
    f = f[:, :, 0]

# Threshold Otsu
h, _ = np.histogram(f, bins=256, range=(0, 255))
To, So = otsuthresh(h)
print(round(255 * To))
go = f > (255 * To)

# Threshold Gradient Laplacian
G = gradlapthresh(f, 1, 0.3)
print(G['G2'] * 255)
print(G['G3'])

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(f, cmap='gray')
axes[0].axis('off')

axes[1].hist(f.ravel(), bins=256, range=(0, 255), color='k')

axes[2].imshow(go, cmap='gray')
axes[2].axis('off')

axes[3].imshow(G['G11'], cmap='gray')
axes[3].axis('off')

vals = G['G16'][G['G16'] != 0]
if vals.size > 0:
    vals_u8 = np.clip(np.round(vals * 255), 0, 255).astype(np.uint8)
    counts, bins = np.histogram(vals_u8, bins=256, range=(0, 256))
    axes[4].plot(bins[:-1], counts, 'k')

axes[5].imshow(G['G1'], cmap='gray')
axes[5].axis('off')

plt.tight_layout()
plt.savefig('Figure1040.png', dpi=300, bbox_inches='tight')
plt.show()
