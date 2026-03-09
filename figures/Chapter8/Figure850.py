from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.fftpack import dct
from libDIP.watermark4e import watermark4e
from libDIPUM.data_path import dip_data

# %% Init
np.random.seed(123)

# %% Parameters
watermark_size = 1000

# %% Data
f = imread(dip_data("lena.tif"))
if f.ndim == 3:
    f = f[..., 0]
f = f.astype(np.uint8)


# %% Watermark sequence
m1 = np.random.randn(watermark_size)
m2 = np.random.randn(watermark_size)


# %% Add watermark
g1, w1 = watermark4e(f, m1)
diff1 = g1.astype(float) - f.astype(float)

g2, w2 = watermark4e(f, m2)
diff2 = g2.astype(float) - f.astype(float)


# %% Verify modified coefficients relationship (coarse check).
def compute_correlation(g: Any, w: Any):
    """compute_correlation."""
    k = np.size(w["m"])

    # dct2 equivalent (orthonormal)
    G = dct(dct(g.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho")

    # MATLAB: [coef, index] = sort(abs(G(:)), 'descend'); coef(1:k)
    coef = np.sort(np.abs(G).ravel())[::-1]
    extracted_coefs = coef[:k]

    original_coefs = np.asarray(w["coef"]).ravel()
    if (
        extracted_coefs.size == 0
        or np.std(extracted_coefs) == 0
        or np.std(original_coefs) == 0
    ):
        return 0.0
    r = np.corrcoef(extracted_coefs, original_coefs)[0, 1]
    return float(r)


r1 = compute_correlation(g1, w1)
r2 = compute_correlation(g2, w2)


# %% Display
fig = plt.figure(1, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(g1, cmap="gray")
plt.title(f"Watermarked, r = {r1:.3g}")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(diff1, cmap="gray")
plt.title(f"Difference (Max) = {np.max(np.abs(diff1)):.6g}")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(g2, cmap="gray")
plt.title(f"Watermarked, r = {r2:.3g}")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(diff2, cmap="gray")
plt.title(f"Difference (Max) = {np.max(np.abs(diff2)):.6g}")
plt.axis("off")

plt.tight_layout()
fig.savefig("Figure850.png", dpi=150, bbox_inches="tight")
plt.show()
