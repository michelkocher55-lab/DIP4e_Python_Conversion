import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve
from libDIPUM.bellmf import bellmf
from libDIPUM.onemf import onemf
from libDIPUM.triangmf import triangmf
from libDIPUM.fuzzysysfcn import fuzzysysfcn
from libDIPUM.approxfcn import approxfcn
from libDIPUM.data_path import dip_data


# MATLAB-like tofloat/revertClass behavior for this script
f_in = imread(dip_data('headCT.tif'))
if f_in.ndim == 3:
    f_in = f_in[:, :, 0]
orig_dtype = f_in.dtype
f = img_as_float(f_in)

def revertClass(x):
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        y = np.clip(x, 0.0, 1.0)
        y = np.round(y * info.max).astype(orig_dtype)
        return y
    return x.astype(orig_dtype)


# The fuzzy system has four inputs: differences with neighbors.
z1 = convolve(f, np.array([[0, -1, 1]], dtype=float), mode='nearest')
z2 = convolve(f, np.array([[0], [-1], [1]], dtype=float), mode='nearest')
z3 = convolve(f, np.array([[1], [-1], [0]], dtype=float), mode='nearest')
z4 = convolve(f, np.array([[1, -1, 0]], dtype=float), mode='nearest')

# Input membership functions.
zero = lambda z: bellmf(z, -0.3, 0)  # noqa: E731
not_used = lambda z: onemf(z)         # noqa: E731

# Output membership functions.
black = lambda z: triangmf(z, 0, 0, 0.75)      # noqa: E731
white = lambda z: triangmf(z, 0.25, 1, 1)      # noqa: E731

# 4 rules x 4 inputs
inmf = [
    [zero,     not_used, zero,     not_used],
    [not_used, not_used, zero,     zero],
    [not_used, zero,     not_used, zero],
    [zero,     zero,     not_used, not_used],
]

# Extra output MF gives automatic else-rule behavior
outmf = [white, white, white, white, black]

# Output range
vrange = [0, 1]

# Build fuzzy system and LUT approximation
F = fuzzysysfcn(inmf, outmf, vrange)
G = approxfcn(F, np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]], dtype=float))
z1c = np.clip(z1, -1.0, 1.0)
z2c = np.clip(z2, -1.0, 1.0)
z3c = np.clip(z3, -1.0, 1.0)
z4c = np.clip(z4, -1.0, 1.0)
gf = G(z1c, z2c, z3c, z4c)

# Convert output back to class of input image
g = revertClass(gf)

# Display 1
fig1 = plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(z1, cmap='gray')
plt.title('d6')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(z2, cmap='gray')
plt.title('d2')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(z3, cmap='gray')
plt.title('d8')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(z4, cmap='gray')
plt.title('d4')
plt.axis('off')

plt.tight_layout()
fig1.savefig('Figure379.png')
print('Saved Figure379.png')

# Display 2
fig2 = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 2, 1)
plt.imshow(f, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
plt.axis('off')

ax2 = plt.subplot(2, 2, 2)
if np.issubdtype(g.dtype, np.integer):
    plt.imshow(g, cmap='gray', vmin=0, vmax=np.iinfo(g.dtype).max)
else:
    plt.imshow(g, cmap='gray', vmin=0, vmax=1)
plt.title('Fuzzy edge enhancement')
plt.axis('off')

ax3 = plt.subplot(2, 2, 3)
gm = gf - np.min(gf)
if np.max(gm) > 0:
    gs = np.uint8(255.0 * (gm / np.max(gm)))
else:
    gs = np.zeros_like(gm, dtype=np.uint8)
plt.imshow(gs, cmap='gray')
plt.title('After scaling')
plt.axis('off')

# MATLAB linkaxes equivalent for image limits consistency
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, f.shape[1] - 1)
    ax.set_ylim(f.shape[0] - 1, 0)

plt.tight_layout()
fig2.savefig('Figure379Bis.png')
print('Saved Figure379Bis.png')

plt.show()
