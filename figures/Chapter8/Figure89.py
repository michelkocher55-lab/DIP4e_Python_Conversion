import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

# Data
f = imread(dip_data('lena.tif'))

# Display
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(f, cmap='gray')
axes[0].axis('off')

counts, edges = np.histogram(f.ravel().astype(float), bins=256)
centers = (edges[:-1] + edges[1:]) / 2
axes[1].bar(centers, counts, width=edges[1] - edges[0])
axes[1].set_aspect('auto')
axes[1].set_box_aspect(1)
axes[1].set_anchor('C')

fig.subplots_adjust(wspace=0.3)
plt.savefig('Figure89.png')
plt.show()
