from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
from scipy.ndimage import correlate


def fspecial(type_filter: Any, *args: Any):
    """
    Mimics MATLAB's fspecial function for 'gaussian'.
    """
    if type_filter == "gaussian":
        hsize = args[0]
        sigma = args[1]

        if isinstance(hsize, (int, float)):
            hsize = (hsize, hsize)

        m, n = hsize
        # Use ogrid ensuring exact shape m, n
        # Center is at (m-1)/2, (n-1)/2
        y, x = np.ogrid[0:m, 0:n]
        y = y - (m - 1) / 2.0
        x = x - (n - 1) / 2.0

        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    else:
        raise NotImplementedError(f"Filter type '{type_filter}' not implemented.")


from scipy.ndimage import correlate1d
from libDIPUM.data_path import dip_data


def imfilter(img: Any, kernel: Any, mode: Any = "constant"):
    """
    Mimics MATLAB's imfilter.
    Uses separable convolution if kernel is rank 1.
    """
    # Map MATLAB modes to scipy.ndimage modes
    if mode == "replicate":
        scipy_mode = "nearest"
    elif mode == "symmetric":
        scipy_mode = "reflect"
    elif mode == "circular":
        scipy_mode = "wrap"
    else:
        scipy_mode = mode  # 'constant'

    # Check for separability using SVD
    # kernel shape (M, N)
    if kernel.ndim == 2:
        try:
            u, s, vh = np.linalg.svd(kernel)
            # Check if rank 1 approx
            # Ratio of first singular value to total energy or second singular value
            if s[1] < 1e-5 * s[0]:
                # Separable
                # k = s[0] * u[:, 0] * vh[0, :]
                # Vertical kernel (u[:, 0] * sqrt(s[0]))
                # Horizontal kernel (vh[0, :] * sqrt(s[0]))

                # Assign sign/magnitude factors
                scale = np.sqrt(s[0])
                k_vert = u[:, 0] * scale
                k_horz = vh[0, :] * scale

                # Convolve columns (axis 0) then rows (axis 1)
                temp = correlate1d(img, k_vert, axis=0, mode=scipy_mode)
                return correlate1d(temp, k_horz, axis=1, mode=scipy_mode)
        except Exception:
            pass  # Fallback to 2D

    # Fallback to standard 2D correlation (slow for large kernels)
    return correlate(img, kernel, mode=scipy_mode)


# Parameters
Sigma = 64
HSize = 512

# Data Loading
img_name = dip_data("checkerboard1024-shaded.tif")
f_orig = imread(img_name)
if f_orig.ndim == 3:
    f_orig = f_orig[:, :, 0]

# Convert to float
f = img_as_float(f_orig)

# Kernel
# h = fspecial ('gaussian', HSize, Sigma);
h = fspecial("gaussian", HSize, Sigma)

# Filtering
# fs = imfilter (f, h, 'replicate');
fs = imfilter(f, h, "replicate")

# Pointwise division
# g = double(f)./double(fs);
epsilon = 1e-6
g = f / (fs + epsilon)

# Thresholding
try:
    thresh = threshold_otsu(g)
    X = g > thresh
except ValueError:
    X = np.zeros_like(g, dtype=bool)

# Display
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
