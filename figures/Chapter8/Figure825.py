import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIP.tmat4e import tmat4e
from libDIPUM.compare import compare
from libDIPUM.data_path import dip_data


def blkdct(x, a):
    return a @ x @ a.conj().T


def blkzero_zonal(x, mask):
    return x * mask


def blkzero_magnitude(x, keep_frac):
    m, n = x.shape
    k = int(np.round(keep_frac * m * n))
    y = np.zeros_like(x)
    if k <= 0:
        return y
    idx = np.argsort(np.abs(x).ravel())[::-1]
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
            out[r:r + block_size, c:c + block_size] = func(fp[r:r + block_size, c:c + block_size])

    if pad_partial_blocks and (pad_m or pad_n):
        out = out[:m, :n]

    return out


def compute_zonal_mask(f, t, block_size, keep_frac):
    # Forward transform block-by-block.
    y = blockproc_square(f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True)

    # Build 3D stack explicitly to avoid reshape-order mismatch with MATLAB.
    m, n = y.shape
    pad_m = (block_size - (m % block_size)) % block_size
    pad_n = (block_size - (n % block_size)) % block_size
    yp = np.pad(y, ((0, pad_m), (0, pad_n)), mode='constant') if (pad_m or pad_n) else y

    nb_r = yp.shape[0] // block_size
    nb_c = yp.shape[1] // block_size
    num_blocks = nb_r * nb_c

    p = np.zeros((block_size, block_size, num_blocks), dtype=float)
    kblk = 0
    for br in range(nb_r):
        for bc in range(nb_c):
            block = yp[
                br * block_size:(br + 1) * block_size,
                bc * block_size:(bc + 1) * block_size,
            ]
            p[:, :, kblk] = np.real(block)
            kblk += 1

    v = np.var(p, axis=2, ddof=0)

    k = int(np.round(keep_frac * block_size * block_size))
    idx = np.argsort(v.ravel())[::-1]
    msk = np.zeros((block_size, block_size), dtype=float)
    if k > 0:
        msk.ravel()[idx[:k]] = 1.0

    return v, msk


def zonal_coding(f, t, block_size, mask):
    y = blockproc_square(f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True)
    y = blockproc_square(y, block_size, lambda b: blkzero_zonal(b, mask), pad_partial_blocks=True)
    y = blockproc_square(y, block_size, lambda b: blkdct(b, t.conj().T), pad_partial_blocks=True)
    return np.real(y)


def magnitude_coding(f, t, block_size, keep_frac):
    y = blockproc_square(f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True)
    y = blockproc_square(y, block_size, lambda b: blkzero_magnitude(b, keep_frac), pad_partial_blocks=True)
    y = blockproc_square(y, block_size, lambda b: blkdct(b, t.conj().T), pad_partial_blocks=True)
    return np.real(y)


# Parameters
keep_frac = 0.125
block_size = 8
t = tmat4e('DCT', block_size)

# Data
f = imread(dip_data('lena.tif')).astype(float)
if f.ndim == 3:
    f = f[..., 0]

# Magnitude coding
f_hat_magnitude = magnitude_coding(f, t, block_size, keep_frac)
e_magnitude = f - f_hat_magnitude
e_rms_magnitude = compare(f, f_hat_magnitude, 0)

# Zonal coding
variances, mask = compute_zonal_mask(f, t, block_size, keep_frac)
f_hat_zonal = zonal_coding(f, t, block_size, mask)
e_zonal = f - f_hat_zonal
e_rms_zonal = compare(f, f_hat_zonal, 0)

# Display figure 1
fig1 = plt.figure(1, figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(f, cmap='gray')
plt.title('Original image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log10(1 + variances), cmap='gray')
plt.title('log(1+variances)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
plt.title(f'Mask, KeepFrac = {keep_frac:g}')
plt.axis('off')

plt.tight_layout()
fig1.savefig('Figure825.png', dpi=150, bbox_inches='tight')

# Display figure 2
fig2 = plt.figure(2, figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(f_hat_magnitude, cmap='gray')
plt.title('Magnitude Coding')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(4 * e_magnitude, cmap='gray')
plt.title(f'4 * error, RMS = {e_rms_magnitude:g}')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(f_hat_zonal, cmap='gray')
plt.title('Zonal Coding')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(4 * e_zonal, cmap='gray')
plt.title(f'4 * error, RMS = {e_rms_zonal:g}')
plt.axis('off')

plt.tight_layout()
fig2.savefig('Figure825Bis.png', dpi=150, bbox_inches='tight')

plt.show()
