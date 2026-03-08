
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.io import imread
from PIL import Image
from libDIPUM.sigmamf import sigmamf
from libDIPUM.triangmf import triangmf
from libDIPUM.data_path import dip_data

print("Running Figure374 (Fuzzy Contrast Enhancement)...")

# Image loading (Exact path)
img_name = dip_data('einstein-low-contrast.tif')
f = imread(img_name)
if f.ndim == 3: f = f[:, :, 0]

if f.ndim == 3: f = f[:,:,0]
f = f.astype(float)

# 1. Histogram Equalization
f_uint8 = f.astype(np.uint8)
g1_float = exposure.equalize_hist(f_uint8)
g1 = (g1_float * 255).astype(np.uint8)

# 2. Fuzzy Enhancement
# Note: sigmamf expects values. If z is 0-255.
udark_val = 1 - sigmamf(f, 74, 127)
ugray_val = triangmf(f, 74, 127, 180)
ubright_val = sigmamf(f, 127, 180)

vd = 0.0
vg = 127.0
vb = 255.0

numerator = udark_val * vd + ugray_val * vg + ubright_val * vb
denominator = udark_val + ugray_val + ubright_val

denominator[denominator == 0] = 1e-6

g2 = numerator / denominator
g2 = np.clip(g2, 0, 255).astype(np.uint8)

# Setup Plot 1 (Images)
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
axes1[0].imshow(f, cmap='gray', vmin=0, vmax=255)
axes1[0].set_title('Original')
axes1[0].axis('off')

axes1[1].imshow(g1, cmap='gray', vmin=0, vmax=255)
axes1[1].set_title('Histogram Equalization')
axes1[1].axis('off')

axes1[2].imshow(g2, cmap='gray', vmin=0, vmax=255)
axes1[2].set_title('Fuzzy Logic Enhancement')
axes1[2].axis('off')

plt.savefig('Figure374.png')

# Setup Plot 2 (Histograms and MFs)
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Hist Original
axes2[0, 0].hist(f.ravel(), bins=256, range=(0, 255), color='k', histtype='stepfilled')
axes2[0, 0].set_title('Hist: Original')
axes2[0, 0].set_xlim([0, 255])

# Hist Equalized
axes2[0, 1].hist(g1.ravel(), bins=256, range=(0, 255), color='k', histtype='stepfilled')
axes2[0, 1].set_title('Hist: Equalized')
axes2[0, 1].set_xlim([0, 255])

# Hist Original with MFs
ax_hist_mf = axes2[1, 0]
counts, bins = np.histogram(f.ravel(), bins=256, range=(0, 255))
counts_norm = counts / (counts.max() + 1e-6)
ax_hist_mf.bar(bins[:-1], counts_norm, width=1, color='gray', alpha=0.5)

# Plot MFs
z = np.linspace(0, 255, 500)
mf_dark = 1 - sigmamf(z, 74, 127)
mf_gray = triangmf(z, 74, 127, 180)
mf_bright = sigmamf(z, 127, 180)

ax_hist_mf.plot(z, mf_dark, 'k-', linewidth=2, label='Dark')
ax_hist_mf.plot(z, mf_gray, 'k--', linewidth=2, label='Gray')
ax_hist_mf.plot(z, mf_bright, 'k:', linewidth=2, label='Bright')
ax_hist_mf.set_title('Hist & MFs')
ax_hist_mf.set_xlim([0, 255])
ax_hist_mf.legend(loc='upper right')

# Hist Fuzzy Enhanced
axes2[1, 1].hist(g2.ravel(), bins=256, range=(0, 255), color='k', histtype='stepfilled')
axes2[1, 1].set_title('Hist: Fuzzy Enhanced')
axes2[1, 1].set_xlim([0, 255])

plt.tight_layout()
plt.savefig('Figure374Bis.png')
print("Saved Figure374.png and Figure374Bis.png")
plt.show()