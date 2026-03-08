import os
import sys
import matplotlib.pyplot as plt
import ia870 as ia
from skimage.io import imread
from libDIPUM.data_path import dip_data

# Parameters
h = 30
bc8 = ia.iasebox()

# Data
image_path = dip_data('Fig1056(a)(blob_original).tif')
f = imread(image_path)
# Marker detection
f3 = ia.iahmin(f, h, bc8)   # HMin filter
m3 = ia.iaregmin(f3, bc8)   # Regional minima

# Morphological gradient
x = ia.iagradm(m3)

# Display
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(2, 2, 1)
ax.imshow(f, cmap='gray')
ax.set_title('Original image')
ax.axis('off')

ax = fig.add_subplot(2, 2, 2)
ax.imshow(f3, cmap='gray')
ax.set_title('After removal of min < 30')
ax.axis('off')

ax = fig.add_subplot(2, 2, 3)
ax.imshow(m3, cmap='gray')
ax.set_title('Regional minimum')
ax.axis('off')

ax = fig.add_subplot(2, 2, 4)
ax.imshow(x, cmap='gray')
ax.set_title('Gradient')
ax.axis('off')

plt.tight_layout()
plt.savefig('Figure1062.png')
print('Saved Figure1062.png')
plt.show()