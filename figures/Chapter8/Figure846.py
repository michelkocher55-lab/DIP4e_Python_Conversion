import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.compare import compare
from libDIPUM.im2jpeg2k import im2jpeg2k
from libDIPUM.imratio import imratio
from libDIPUM.jpeg2k2im import jpeg2k2im
from libDIPUM.data_path import dip_data


def imcrop_matlab(img, rect):
    # MATLAB: rect = [x, y, w, h], inclusive integer rectangle.
    x, y, w, h = [int(v) for v in rect]
    return img[y:y + h + 1, x:x + w + 1]


# Parameters
mu_b = [8, 8, 8, 8]
epsilon_b = [8.5, 7, 6.5, 6.0]
n_level = 5

# Data
f = imread(dip_data('lena.tif'))
if f.ndim == 3:
    f = f[..., 0]
if f.dtype != np.uint8:
    f = np.clip(np.round(f), 0, 255).astype(np.uint8)

# Process
f_hat = []
compression_ratio = []
rmse = []

for i in range(len(mu_b)):
    q = [mu_b[i], epsilon_b[i]]
    y, _ = im2jpeg2k(f, n_level, q)
    compression_ratio.append(imratio(f, y))

    rec = jpeg2k2im(y)
    f_hat.append(rec)
    rmse.append(compare(f.astype(float), rec.astype(float), 0))

# Display
fig = plt.figure(1, figsize=(12, 14))
for i in range(len(mu_b)):
    plt.subplot(4, 3, 1 + i * 3)
    plt.imshow(f_hat[i], cmap='gray')
    plt.title(f'RMSE = {rmse[i]:.2g} Comp. = {compression_ratio[i]:.2g}')
    plt.axis('off')

    plt.subplot(4, 3, 2 + i * 3)
    plt.imshow(f.astype(float) - f_hat[i].astype(float), cmap='gray')
    plt.title('error')
    plt.axis('off')

    plt.subplot(4, 3, 3 + i * 3)
    temp = imcrop_matlab(f_hat[i], [243 - 21, 249 - 21, 64, 64])
    plt.imshow(temp, cmap='gray')
    plt.title('zoom')
    plt.axis('off')

plt.tight_layout()
fig.savefig('Figure846.png', dpi=150, bbox_inches='tight')
plt.show()
