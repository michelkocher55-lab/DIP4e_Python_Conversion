import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from fig81bc import fig81bc
from libDIPUM.data_path import dip_data

# Figure81

# Data
fa = imread(dip_data('Fig0801(a).tif'))
fb = fig81bc('b')
fc = fig81bc('c')

# Display (Figure 1)
fig1, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(fa, cmap='gray', vmin=0, vmax=255)
axes[0].axis('off')
axes[1].imshow(fb, cmap='gray', vmin=0, vmax=255)
axes[1].axis('off')
axes[2].imshow(fc, cmap='gray', vmin=0, vmax=255)
axes[2].axis('off')
fig1.tight_layout()

# Display (Figure 2): bar(hist(double(fb(:)), 256))
fig2, ax2 = plt.subplots(figsize=(8, 4))
counts, edges = np.histogram(fb.astype(float).ravel(), bins=256)
centers = 0.5 * (edges[:-1] + edges[1:])
ax2.bar(centers, counts, width=(edges[1] - edges[0]))
fig2.tight_layout()

# Print to file
fig1.savefig('Figure81.png', dpi=300, bbox_inches='tight')
fig2.savefig('Figure82.png', dpi=300, bbox_inches='tight')

plt.show()
