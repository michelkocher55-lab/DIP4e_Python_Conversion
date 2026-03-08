import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.im2jpeg import im2jpeg
from libDIPUM.jpeg2im import jpeg2im
from libDIPUM.imratio import imratio
from libDIPUM.compare import compare
from libDIPUM.data_path import dip_data


# Parameters
quality = [4, 8]

# Data
f = imread(dip_data('lena.tif'))
if f.ndim == 3:
    f = f[..., 0]
f = f.astype(np.uint8)

# Compression, decompression
f_hat = []
compression_ratio = []
rmse = []
e = []
error_min = []
error_max = []

for q in quality:
    y = im2jpeg(f, q)
    fh = jpeg2im(y)
    f_hat.append(fh)
    compression_ratio.append(imratio(f, y))
    rmse.append(compare(f, fh, 0))

    err = f.astype(np.float64) - fh.astype(np.float64)
    e.append(err)
    error_min.append(np.min(err))
    error_max.append(np.max(err))


# MATLAB imcrop([x, y, w, h]) equivalent (inclusive width/height)
def imcrop_matlab(img, rect):
    x, y, w, h = [int(v) for v in rect]
    x2 = x + w + 1
    y2 = y + h + 1
    return img[y:y2, x:x2]


# Display
fig = plt.figure(1, figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(f_hat[0], cmap='gray', vmin=0, vmax=255)
plt.title(
    f"Q = {quality[0]}, RMSE = {rmse[0]:.2g} Comp. = {compression_ratio[0]:.2g}"
)
plt.axis('off')

plt.subplot(2, 3, 2)
# Matches MATLAB script: uses ErrorMin(2), ErrorMax(2) for first error display.
plt.imshow(e[0], cmap='gray', vmin=error_min[1], vmax=error_max[1])
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(imcrop_matlab(f_hat[0], [234, 250, 60, 40]), cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(f_hat[1], cmap='gray', vmin=0, vmax=255)
plt.title(
    f"Q = {quality[1]}, RMSE = {rmse[1]:.2g} Comp. = {compression_ratio[1]:.2g}"
)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(e[1], cmap='gray', vmin=error_min[1], vmax=error_max[1])
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(imcrop_matlab(f_hat[1], [234, 250, 60, 40]), cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
fig.savefig('Figure829.png', dpi=150, bbox_inches='tight')
plt.show()
