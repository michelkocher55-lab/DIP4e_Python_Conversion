import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.ntrop import ntrop
from libDIPUM.data_path import dip_data

# Data
f_raw = imread(dip_data('nasaframe78.tif'))
f_next_raw = imread(dip_data('nasaframe79.tif'))
if f_raw.ndim == 3:
    f_raw = f_raw[..., 0]
if f_next_raw.ndim == 3:
    f_next_raw = f_next_raw[..., 0]

# TIFF input is video-inversed for this dataset; invert polarity.
if np.issubdtype(f_raw.dtype, np.integer):
    f = (np.iinfo(f_raw.dtype).max - f_raw).astype(float)
else:
    f = (np.max(f_raw) - f_raw).astype(float)

if np.issubdtype(f_next_raw.dtype, np.integer):
    f_next = (np.iinfo(f_next_raw.dtype).max - f_next_raw).astype(float)
else:
    f_next = (np.max(f_next_raw) - f_next_raw).astype(float)

# Time linear predictive coding
delta = f - f_next

# Display
fig = plt.figure(1, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(f, cmap='gray')
plt.title('Frame 78')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(f_next, cmap='gray')
plt.title('Frame 79')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(delta, cmap='gray')
plt.title('Delta')
plt.axis('off')

plt.subplot(2, 2, 4)
hd, _ = np.histogram(delta.ravel(), bins=256)
plt.bar(np.arange(hd.size), hd)
sigma = np.std(delta, ddof=1)  # MATLAB std2 equivalent normalization (N-1)
plt.title(f'H = {ntrop(delta.ravel()):g}, $\sigma$ = {sigma:g}')

plt.tight_layout()
fig.savefig('Figure832.png', dpi=150, bbox_inches='tight')
plt.show()
