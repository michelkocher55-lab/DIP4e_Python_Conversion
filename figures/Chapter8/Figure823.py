import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIP.tmat4e import tmat4e
from libDIPUM.compare import compare
from libDIPUM.data_path import dip_data


# Figure 823


def blkdct(x, a):
    """Block transform using matrix multiplications."""
    return a @ x @ a.conj().T


def blkzero(x, keep_frac):
    """Keep the largest-magnitude coefficients in a block."""
    n = x.size
    k = int(np.round(keep_frac * n))

    idx = np.argsort(np.abs(x).ravel())[::-1]
    y = np.zeros_like(x)
    if k > 0:
        y.ravel()[idx[:k]] = x.ravel()[idx[:k]]
    return y


def blockproc_square(f, block_size, func):
    """Apply func to non-overlapping square blocks with zero-padding of partial blocks."""
    m, n = f.shape
    pad_m = (block_size - (m % block_size)) % block_size
    pad_n = (block_size - (n % block_size)) % block_size

    fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode='constant')
    out = np.zeros_like(fp, dtype=np.complex128)

    mp, np_ = fp.shape
    for r in range(0, mp, block_size):
        for c in range(0, np_, block_size):
            out[r:r + block_size, c:c + block_size] = func(fp[r:r + block_size, c:c + block_size])

    # Crop back to original size for fair RMS comparison.
    return out[:m, :n]


def process(f, t, block_size, keep_frac):
    # Compute forward transform of block_size x block_size blocks.
    y = blockproc_square(f, block_size, lambda b: blkdct(b, t))

    # Zero coefficients based on magnitude.
    y = blockproc_square(y, block_size, lambda b: blkzero(b, keep_frac))

    # Compute inverse transform.
    y = np.real(blockproc_square(y, block_size, lambda b: blkdct(b, np.conj(t.T))))

    # Compute RMS error.
    return compare(f, y, 0)


# Parameters
BlockSize = [2, 4, 8, 16, 32, 64, 128, 256, 512]
KeepFrac = 0.25  # 75% are zeroed

# Data
f = imread(dip_data('lena.tif')).astype(float)
if f.ndim == 3:
    f = f[..., 0]

# Process
RMS = np.zeros((3, len(BlockSize)), dtype=float)
transforms = [[None for _ in BlockSize] for _ in range(3)]

for iter_idx in range(3):
    for iter1, bsz in enumerate(BlockSize):
        if iter_idx == 0:
            transforms[iter_idx][iter1] = tmat4e('DFT', bsz)
        elif iter_idx == 1:
            transforms[iter_idx][iter1] = tmat4e('DCT', bsz)
        else:
            transforms[iter_idx][iter1] = tmat4e('WHT', bsz)

        RMS[iter_idx, iter1] = process(f, transforms[iter_idx][iter1], bsz, KeepFrac)

# Display
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(BlockSize, RMS[0, :], 'k-s', label='DFT')
ax.plot(BlockSize, RMS[1, :], 'k--o', label='DCT')
ax.plot(BlockSize, RMS[2, :], 'k:d', label='WHT')
ax.legend()
ax.set_xlabel('Block size')
ax.set_ylabel('RMS error')
ax.set_title(f'Only {KeepFrac * 100:g} % of the coefficients are kept')
ax.autoscale(enable=True, axis='both', tight=True)

plt.tight_layout()
fig.savefig('Figure823.png', dpi=300, bbox_inches='tight')
plt.show()
