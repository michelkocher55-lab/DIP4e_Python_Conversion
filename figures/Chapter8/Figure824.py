import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIP.tmat4e import tmat4e
from libDIPUM.data_path import dip_data


def blkdct(x, a):
    return a @ x @ a.conj().T


def blkzero(x, keep_frac):
    nr, nc = x.shape
    k = int(np.round(keep_frac * nr * nc))
    y = np.zeros_like(x)
    idx = np.argsort(np.abs(x).ravel())[::-1]
    if k > 0:
        y.ravel()[idx[:k]] = x.ravel()[idx[:k]]
    return y


def blockproc_square(f, block_size, func, pad_partial_blocks=True):
    m, n = f.shape
    if pad_partial_blocks:
        pad_m = (block_size - (m % block_size)) % block_size
        pad_n = (block_size - (n % block_size)) % block_size
        fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode='constant')
    else:
        fp = f
        pad_m = 0
        pad_n = 0

    out = np.zeros_like(fp, dtype=np.complex128)
    mp, np_ = fp.shape

    for r in range(0, mp, block_size):
        for c in range(0, np_, block_size):
            out[r:r + block_size, c:c + block_size] = func(
                fp[r:r + block_size, c:c + block_size]
            )

    if pad_partial_blocks and (pad_m or pad_n):
        out = out[:m, :n]

    return out


def process(f, t, block_size, keep_frac):
    y = blockproc_square(f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True)
    y = blockproc_square(y, block_size, lambda b: blkzero(b, keep_frac), pad_partial_blocks=True)
    y = blockproc_square(y, block_size, lambda b: blkdct(b, t.conj().T), pad_partial_blocks=True)
    return np.real(y)


def imcrop_matlab(img, rect):
    # MATLAB imcrop([x, y, w, h]) includes both endpoints for integer coords.
    x, y, w, h = [int(v) for v in rect]
    return img[y:y + h + 1, x:x + w + 1]


# Parameters
keep_frac = 0.25
block_sizes = [2, 4, 8]
t = [tmat4e('DCT', bs) for bs in block_sizes]

# Data
f = imread(dip_data('lena.tif')).astype(float)
if f.ndim == 3:
    f = f[..., 0]

f = imcrop_matlab(f, [243 - 7, 249 - 7, 31, 31])

# Process
f_hat = [process(f, t[i], block_sizes[i], keep_frac) for i in range(len(block_sizes))]

# Display
fig = plt.figure(1, figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(f, cmap='gray')
plt.title('Original')
plt.axis('off')

for i, bs in enumerate(block_sizes):
    plt.subplot(1, 4, i + 2)
    plt.imshow(f_hat[i], cmap='gray')
    plt.title(f'Block Size = {bs}')
    plt.axis('off')

plt.tight_layout()
fig.savefig('Figure824.png', dpi=150, bbox_inches='tight')
plt.show()
