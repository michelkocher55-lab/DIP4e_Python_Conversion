import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import correlate1d
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data('testpattern4096.tif')
f = imread(img_name)
if f.ndim == 3:
    f = f[:, :, 0]
f = img_as_float(f)


def gaussian_1d(size, sigma):
    radius = (size - 1) // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    w /= np.sum(w)
    return w


# Separable kernels equivalent to normalized 2D Gaussian kernels
w187 = gaussian_1d(187, 31.0)
w745 = gaussian_1d(745, 124.0)

# Filtering (MATLAB imfilter(..., 'symmetric') -> mode='reflect')
g187 = correlate1d(f, w187, axis=1, mode='reflect')
g187 = correlate1d(g187, w187, axis=0, mode='reflect')

g745 = correlate1d(f, w745, axis=1, mode='reflect')
g745 = correlate1d(g745, w745, axis=0, mode='reflect')

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(f, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(g187, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(g745, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.tight_layout()
plt.savefig('Figure346.png')
print('Saved Figure346.png')
plt.show()
