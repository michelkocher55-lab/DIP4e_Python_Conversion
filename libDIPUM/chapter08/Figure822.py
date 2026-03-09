from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIP.tmat4e import tmat4e
from libDIPUM.data_path import dip_data


def blkdct(x: Any, a: Any):
    """blkdct."""
    return a @ x @ a.conj().T


def blkzero(x: Any):
    """blkzero."""
    k = 32
    y = np.zeros_like(x)
    idx = np.argsort(np.abs(x).ravel())[::-1]
    y.ravel()[idx[:k]] = x.ravel()[idx[:k]]
    return y


def blockproc_8x8(f: Any, func: Any, pad_partial_blocks: Any = True):
    """blockproc_8x8."""
    m, n = f.shape
    if pad_partial_blocks:
        pad_m = (8 - (m % 8)) % 8
        pad_n = (8 - (n % 8)) % 8
        fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode="constant")
    else:
        fp = f

    out = np.zeros_like(fp, dtype=np.complex128)
    mp, np_ = fp.shape

    for r in range(0, mp, 8):
        for c in range(0, np_, 8):
            out[r : r + 8, c : c + 8] = func(fp[r : r + 8, c : c + 8])

    return out


def process(f: Any, t: Any):
    """process."""
    y = blockproc_8x8(f, lambda b: blkdct(b, t), pad_partial_blocks=True)
    y = blockproc_8x8(y, blkzero, pad_partial_blocks=True)
    y = blockproc_8x8(y, lambda b: blkdct(b, t.conj().T), pad_partial_blocks=True)
    return np.real(y)


# Parameters
t = [
    tmat4e("DFT", 8),
    tmat4e("DCT", 8),
    tmat4e("DHT", 8),
]

# Data
f = imread(dip_data("lena.tif")).astype(float)
if f.ndim == 3:
    f = f[..., 0]

# Process
f_hat = [process(f, t[0]), process(f, t[1]), process(f, t[2])]

# Display
fig = plt.figure(1, figsize=(12, 7))

plt.subplot(2, 3, 1)
plt.imshow(f_hat[0], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(f_hat[1], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(f_hat[2], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(f - f_hat[0], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(f - f_hat[1], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(f - f_hat[2], cmap="gray")
plt.axis("off")

plt.tight_layout()
fig.savefig("Figure822.png", dpi=150, bbox_inches="tight")
plt.show()
