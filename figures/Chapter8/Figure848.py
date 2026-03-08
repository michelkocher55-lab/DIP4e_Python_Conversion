import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.im2jpeg import im2jpeg
from libDIPUM.jpeg2im import jpeg2im
from libDIPUM.data_path import dip_data


# %% Figure848

# %% Parameters
row = 400
col = 200

# %% Data
f = imread(dip_data('lena.tif'))
if f.ndim == 3:
    f = f[..., 0]
f = f.astype(np.uint8)

# %% Watermark
w = imread(dip_data('Fig0850(a).tif'))
if w.ndim == 3:
    w = w[:, :, 0]
w = w.astype(np.uint8)

# %% Code it in 2 LSB bits from 0 to 3
w = np.bitwise_and(w, 3)

# %% Place it into a 512 by 512 image
temp = np.zeros_like(f, dtype=np.uint8)
rr0 = row - 1
cc0 = col - 1
temp[rr0:rr0 + 52, cc0:cc0 + 153] = w

# %% Fragile watermarking
fw = ((f // 4) * 4 + temp).astype(np.uint8)

# %% Decoding
w_hat = np.bitwise_and(fw, 3)

# %% Attack by using JPEG compression and decompression
fw_attack = jpeg2im(im2jpeg(fw))
w_attack_hat = np.bitwise_and(fw_attack.astype(np.uint8), 3)

# %% Display
fig = plt.figure(1, figsize=(13, 8))

plt.subplot(2, 3, 1)
plt.imshow(f, cmap='gray')
plt.title('Original image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(w, cmap='gray', vmin=0, vmax=3)
plt.title('Watermark on 2 bits')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(fw, cmap='gray')
plt.title('Watermarked image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(w_hat, cmap='gray', vmin=0, vmax=3)
plt.title('Watermark detected')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(fw_attack, cmap='gray')
plt.title('Watermarked image attacked')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(w_attack_hat, cmap='gray', vmin=0, vmax=3)
plt.title('Watermark detected after attack')
plt.axis('off')

plt.tight_layout()
fig.savefig('Figure848.png', dpi=150, bbox_inches='tight')
plt.show()
