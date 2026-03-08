import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.compare import compare
from libDIPUM.lpc2mat import lpc2mat
from libDIPUM.mat2lpc import mat2lpc
from libDIPUM.ntrop import ntrop
from libDIPUM.data_path import dip_data

# Data
f_raw = imread(dip_data('nasaframe67.tif'))
if f_raw.ndim == 3:
    f_raw = f_raw[..., 0]

# TIFF input is video-inversed for this dataset; invert polarity.
if np.issubdtype(f_raw.dtype, np.integer):
    f = (np.iinfo(f_raw.dtype).max - f_raw).astype(float)
else:
    f = (np.max(f_raw) - f_raw).astype(float)

# Predictive coding 1D
y = mat2lpc(f, 1)
f_hat = lpc2mat(y, 1)
rmse = compare(f, f_hat, 0)

# Display
fig = plt.figure(1, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(f, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
hf, _ = np.histogram(f.ravel(), bins=256)
plt.bar(np.arange(hf.size), hf)
plt.title(f'H = {ntrop(f.ravel()):g}')

plt.subplot(2, 2, 3)
plt.imshow(y, cmap='gray')
plt.title(f'RMSE = {rmse:g}')
plt.axis('off')

plt.subplot(2, 2, 4)
hy, _ = np.histogram(y.ravel(), bins=256)
plt.bar(np.arange(hy.size), hy)
plt.title(f'H = {ntrop(y.ravel()):g}')

plt.tight_layout()
fig.savefig('Figure831.png', dpi=150, bbox_inches='tight')
plt.show()
