
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

print("Running Figure428 (Blown IC and Spectrum)...")

# Image loading
img_path = dip_data('blown_ic.tif')
f = imread(img_path)
if f.ndim == 3: f = f[:,:,0]

f = img_as_float(f)

# Fourier transform
# F = fft2(f);
F = np.fft.fft2(f)

# S = abs(F);
S = np.abs(F)

# S = fftshift(S);
S = np.fft.fftshift(S)

# S = log10(1+S);
S_log = np.log10(1 + S)

# Scaling for display
# S = S - min(S(:));
# S = S/max(S(:));
S_disp = S_log - np.min(S_log)
if np.max(S_disp) > 0:
    S_disp = S_disp / np.max(S_disp)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(S_disp, cmap='gray')
axes[1].set_title('Fourier Spectrum')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Figure428.png')
print("Saved Figure428.png")
plt.show()
