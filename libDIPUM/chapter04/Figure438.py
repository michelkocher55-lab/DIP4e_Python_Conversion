from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import correlate
from libDIPUM.paddedsize import paddedsize
from libDIPUM.dftfilt import dftfilt
from helpers.freqz2_equal import freqz2_equal
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("building-600by600.tif")
f = img_as_float(imread(img_path))
NR, NC = f.shape

# Padding
PQ = paddedsize(f.shape)

# Fourier transform
F = np.fft.fft2(f)

# Impulse response (Sobel) and frequency response
h = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)

# MATLAB freqz2 (equal spacing) implementation
H = freqz2_equal(h, PQ[0], PQ[1])
H_disp = H

# Filtering in the spatial domain (imfilter default is correlation)
gs = correlate(f, h, mode="constant", cval=0.0)

# Filtering in the frequency domain
H1 = np.fft.ifftshift(H_disp)
gf = dftfilt(f, H1)


# Display helpers (MATLAB imshow(..., []) autoscale)
def autoscale(img: Any):
    """autoscale."""
    img = np.asarray(img, dtype=float)
    img = img - img.min()
    maxv = img.max()
    if maxv > 0:
        img = img / maxv
    return img


fig = plt.figure(figsize=(10, 8))

# 3D surface of imag(H) subsampled
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
step = 25
H_sub = np.imag(H_disp)[::step, ::step]
x = np.arange(0, H.shape[1], step)
y = np.arange(0, H.shape[0], step)
X, Y = np.meshgrid(x, y, indexing="ij")
ax1.plot_surface(X, Y, H_sub, cmap="gray", linewidth=0, antialiased=False)
ax1.view_init(elev=24, azim=146)
ax1.set_axis_off()

# imag(H)
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(autoscale(np.imag(H_disp)), cmap="gray")
ax2.set_title("H(f_x, f_y)")
ax2.axis("off")

# Spatial domain
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(autoscale(gs), cmap="gray")
ax3.set_title("filtering in the spatial domain")
ax3.axis("off")

# Frequency domain
ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(autoscale(gf), cmap="gray")
ax4.set_title("filtering in the frequency domain")
ax4.axis("off")

plt.tight_layout()
plt.savefig("Figure438.png")
plt.show()
