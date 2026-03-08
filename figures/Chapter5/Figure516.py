
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.cnotch import cnotch
from libDIPUM.imnoise3 import imnoise3
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data('astronaut.tif')
f_orig = imread(img_name)
if f_orig.ndim == 3: f_orig = f_orig[:,:,0] # Use one channel if color
f = img_as_float(f_orig)

M, N = f.shape

# Generate sinusoidal noise pattern
# [r, R, S] = imnoise3(M, N, [25 25], 0.3);
# Note: MATLAB [25 25] means C = [25 25].
# 0.3 is Amplitude A.
r, R, S = imnoise3(M, N, [[25, 25]], A=[0.3])

# Noisy image
g = f + r

# Scaling for display implies mapping range.
# intScaling4e handles this.
gs = intScaling4e(g)

# Spectrum
G_complex = np.fft.fft2(g)
G = np.fft.fftshift(np.abs(G_complex))
Glog = intScaling4e(1 + np.log(G + 1e-9)) # Avoid log(0)

# Create notch filters
impulse_loc = [M//2 + 25, N//2 + 25]
H = cnotch('ideal', 'reject', M, N, [impulse_loc], 2)

# Hc = intScaling4e(fftshift(H));
# cnotch returns uncentered.
Hc = intScaling4e(np.fft.fftshift(H))

# Filter the image
F_g = np.fft.fft2(g)
G_filtered = F_g * H
gf_raw = np.real(np.fft.ifft2(G_filtered))
gf = intScaling4e(gf_raw)

# Eliminated noise (what filtering removed)
noise_eliminated = g - gf_raw
noise_eliminated_s = intScaling4e(noise_eliminated)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(gs, cmap='gray') # Display scaled noisy image
axes[0, 0].set_title('Noisy Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(Glog, cmap='gray')
axes[0, 1].set_title('Spectrum of Noisy Image')
axes[0, 1].axis('off')

axes[1, 0].imshow(Hc, cmap='gray')
axes[1, 0].set_title('Notch Filter')
axes[1, 0].axis('off')

axes[1, 1].imshow(gf, cmap='gray')
axes[1, 1].set_title('Filtered Image')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure516.png')
print("Saved Figure516.png")


# Display eliminated noise in separate figure
plt.figure(figsize=(6, 6))
plt.imshow(noise_eliminated_s, cmap='gray')
plt.title('Eliminated Noise')
plt.axis('off')
plt.tight_layout()
plt.savefig('Figure517.png')
print("Saved Figure517.png")
plt.show()
