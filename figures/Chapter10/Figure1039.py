import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import dilation, square
from libDIPUM.gradlapthresh import gradlapthresh
from libDIPUM.data_path import dip_data

# Figure 10.39
# Using edge features to determine threshold.

# Data
f = imread(dip_data('Fig1041(a)(septagon_small_noisy_mean_0_stdv_10).tif'))
if f.ndim == 3:
    f = f[:, :, 0]

# Gradient and Laplacian with threshold
G = gradlapthresh(f, 0.3, 1)
print(round(255 * G['G2']))

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(f, cmap='gray')
axes[0].axis('off')

axes[1].hist(f.ravel(), bins=256, range=(0, 255), color='k')
axes[1].set_box_aspect(1)

axes[2].imshow(G['G6'], cmap='gray')
axes[2].axis('off')

g16d = dilation((255 * G['G16']).astype(np.uint8), square(3))
axes[3].imshow(g16d, cmap='gray')
axes[3].axis('off')

axes[4].plot(G['G4'], 'k')
axes[4].set_box_aspect(1)

axes[5].imshow(G['G1'], cmap='gray')
axes[5].axis('off')

plt.tight_layout()
plt.savefig('Figure1039.png', dpi=300, bbox_inches='tight')
plt.show()
