import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.compare import compare
from libDIPUM.lpc2mat2d import lpc2mat2d
from libDIPUM.mat2lpc2d import mat2lpc2d
from libDIPUM.ntrop import ntrop
from libDIPUM.data_path import dip_data

# Data
f = imread(dip_data('lena.tif'))
if f.ndim == 3:
    f = f[..., 0]

# Predictive coding 2D
y1 = mat2lpc2d(f, 0.97, 0, 0)
f_hat1 = lpc2mat2d(y1, 0.97, 0, 0)
rmse1 = compare(f.astype(float), f_hat1, 0)

y2 = mat2lpc2d(f, 0.5, 0.5, 0)
f_hat2 = lpc2mat2d(y2, 0.5, 0.5, 0)
rmse2 = compare(f.astype(float), f_hat2, 0)

y3 = mat2lpc2d(f, 0.75, 0.75, -0.5)
f_hat3 = lpc2mat2d(y3, 0.75, 0.75, -0.5)
rmse3 = compare(f.astype(float), f_hat3, 0)

# Display
fig = plt.figure(1, figsize=(10, 7))

plt.subplot(2, 3, 1)
plt.imshow(y1, cmap='gray')
plt.title(f'RMSE = {rmse1:g}')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(y2, cmap='gray')
plt.title(f'RMSE = {rmse2:g}')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(y3, cmap='gray')
plt.title(f'RMSE = {rmse3:g}')
plt.axis('off')

plt.subplot(2, 3, 4)
hist1, _ = np.histogram(y1.ravel(), bins=256)
plt.bar(np.arange(hist1.size), hist1)
plt.title(f'H = {ntrop(y1.astype(np.int8)):g}')

plt.subplot(2, 3, 5)
hist2, _ = np.histogram(y2.ravel(), bins=256)
plt.bar(np.arange(hist2.size), hist2)
plt.title(f'H = {ntrop(y2.astype(np.int8)):g}')

plt.subplot(2, 3, 6)
hist3, _ = np.histogram(y3.ravel(), bins=256)
plt.bar(np.arange(hist3.size), hist3)
plt.title(f'H = {ntrop(y3.astype(np.int8)):g}')

plt.tight_layout()
fig.savefig('Figure840.png', dpi=150, bbox_inches='tight')
plt.show()
