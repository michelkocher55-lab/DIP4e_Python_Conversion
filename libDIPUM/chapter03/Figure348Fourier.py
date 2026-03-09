from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
from scipy.signal import fftconvolve
from libDIPUM.data_path import dip_data


def fspecial(type_filter: Any, *args: Any):
    """
    Mimics MATLAB's fspecial function for 'gaussian'.
    """
    if type_filter == "gaussian":
        hsize = args[0]
        sigma = args[1]

        if isinstance(hsize, (int, float)):
            hsize = (int(hsize), int(hsize))

        m, n = hsize
        y, x = np.ogrid[
            -(m - 1) // 2 : (m - 1) // 2 + 1, -(n - 1) // 2 : (n - 1) // 2 + 1
        ]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    raise NotImplementedError(f"Filter type '{type_filter}' not implemented.")


def imfilter(img: Any, kernel: Any, mode: Any = "constant"):
    """
    Mimics MATLAB's imfilter.
    Uses FFT-based convolution for speed with large kernels.
    """
    if mode == "replicate":
        pad_mode = "edge"
    elif mode == "symmetric":
        pad_mode = "reflect"
    elif mode == "circular":
        pad_mode = "wrap"
    else:
        pad_mode = "constant"

    kh, kw = kernel.shape

    # For output size preservation with mode='valid':
    # need total padding kh-1 and kw-1, split asymmetrically when even.
    pad_top = (kh - 1) // 2
    pad_bottom = (kh - 1) - pad_top
    pad_left = (kw - 1) // 2
    pad_right = (kw - 1) - pad_left

    if pad_mode == "constant":
        padded_img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=pad_mode,
            constant_values=0,
        )
    else:
        padded_img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=pad_mode,
        )

    output = fftconvolve(padded_img, kernel, mode="valid")
    return np.real(output)


print("Running Figure348 (Shading Correction)...")

Sigma = 64
HSize = 512

img_name = dip_data("checkerboard1024-shaded.tif")
f_orig = imread(img_name)
if f_orig.ndim == 3:
    f_orig = f_orig[:, :, 0]

f = img_as_float(f_orig)

h = fspecial("gaussian", HSize, Sigma)

fs = imfilter(f, h, "replicate")

epsilon = 1e-6
g = f / (fs + epsilon)

try:
    thresh = threshold_otsu(g)
    X = g > thresh
except ValueError:
    X = np.zeros_like(g, dtype=bool)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title(f"Original Image f\nSize={f.shape}")
axes[0, 0].axis("off")

axes[0, 1].imshow(fs, cmap="gray")
axes[0, 1].set_title(f"Smoothed Image fs\nSigma={Sigma}, Size={HSize}")
axes[0, 1].axis("off")

axes[1, 0].imshow(g, cmap="gray")
axes[1, 0].set_title("Shading Corrected g = f/fs")
axes[1, 0].axis("off")

axes[1, 1].imshow(X, cmap="gray")
axes[1, 1].set_title("Otsu Thresholded (g)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure348.png")
print("Saved Figure348.png")
plt.show()
