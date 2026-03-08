import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.golomb import golomb
from libDIPUM.mat2huff import mat2huff
from libDIPUM.imratio import imratio
from libDIPUM.data_path import dip_data

# Figure 811

# Parameters
m = 5

# Data
# Read image and subtract the average value.
i = imread(dip_data('Fig81c.tif'))
x = i.astype(float) - 128

# Compute histogram between min and max with bin size 1.
xmin = int(np.min(x))
xmax = int(np.max(x))
x = x.ravel()

edges = np.arange(xmin, xmax + 2)
h, _ = np.histogram(x, bins=edges)
hx = np.linspace(xmin, xmax, xmax - xmin + 1)

# Golomb coding
h1, x1, cr = golomb(x, m)
print(cr)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(hx, h/np.sum(h), 'k-s')
axes[0].set_xticks(np.arange(xmin, xmax + 1, 1))
axes[0].set_box_aspect(1)

x2 = np.linspace(0, xmax - xmin + 1, xmax - xmin + 2)
axes[1].plot(x2, h1/np.sum(h1), 'k-s')
axes[1].set_xticks(np.arange(0, xmax - xmin + 3, 1))
axes[1].set_box_aspect(1)

plt.tight_layout()

# To compute the Huffman alternative ratio
c = mat2huff(i)
cr1 = imratio(i, c)
print(cr1)

# Print to file
fig.savefig('Figure811.png', dpi=300, bbox_inches='tight')

plt.show()
