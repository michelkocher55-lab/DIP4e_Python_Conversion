from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIP.tmat4e import tmat4e
from libDIPUM.compare import compare
from libDIPUM.data_path import dip_data


def compute_q(q: Any):
    """compute_q."""
    base = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=float,
    )
    return q * base


def blkdct(x: Any, a: Any):
    """blkdct."""
    return a @ x @ a.conj().T


def blkidct(x: Any, a: Any, qmatrix: Any):
    """blkidct."""
    x = qmatrix * x
    return a @ x @ a.conj().T


def threshold_coding(x: Any, qmatrix: Any):
    """threshold_coding."""
    return np.round(x / qmatrix)


def blockproc_square(
    f: Any, block_size: Any, func: Any, pad_partial_blocks: Any = True
):
    """blockproc_square."""
    m, n = f.shape
    if pad_partial_blocks:
        pad_m = (block_size - (m % block_size)) % block_size
        pad_n = (block_size - (n % block_size)) % block_size
        fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode="constant")
    else:
        fp = f
        pad_m = 0
        pad_n = 0

    out = np.zeros_like(fp, dtype=np.complex128)
    mp, np_ = fp.shape

    for r in range(0, mp, block_size):
        for c in range(0, np_, block_size):
            out[r : r + block_size, c : c + block_size] = func(
                fp[r : r + block_size, c : c + block_size]
            )

    if pad_partial_blocks and (pad_m or pad_n):
        out = out[:m, :n]

    return out


def process(f: Any, t: Any, block_size: Any, qmatrix: Any):
    """process."""
    y = blockproc_square(f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True)
    y = blockproc_square(
        y, block_size, lambda b: threshold_coding(b, qmatrix), pad_partial_blocks=True
    )
    y = blockproc_square(
        y,
        block_size,
        lambda b: blkidct(b, t.conj().T, qmatrix),
        pad_partial_blocks=True,
    )
    return np.real(y)


# Parameters
q_values = [1, 2, 4, 8, 16, 32]
block_size = 8
t = tmat4e("DCT", block_size)

# Data
f = imread(dip_data("lena.tif")).astype(float)
if f.ndim == 3:
    f = f[..., 0]

# Threshold coding
f_hat_threshold = []
error_rms_threshold = []

for q in q_values:
    q_matrix = compute_q(q)
    f_hat = process(f, t, block_size, q_matrix)
    f_hat_threshold.append(f_hat)
    error_rms_threshold.append(compare(f, f_hat, 0))

# Display
fig = plt.figure(1, figsize=(12, 7))
for i, q in enumerate(q_values):
    plt.subplot(2, 3, i + 1)
    plt.imshow(f_hat_threshold[i], cmap="gray")
    plt.title(f"Thr Cod., Q = {q}, e_{{RMS}} = {error_rms_threshold[i]:.2g}")
    plt.axis("off")

plt.tight_layout()
fig.savefig("Figure828.png", dpi=150, bbox_inches="tight")
plt.show()
