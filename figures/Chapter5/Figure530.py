import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from libDIP.motionBlurTF4e import motionBlurTF4e
from libDIP.constrainedLsTF4e import constrainedLsTF4e
from libDIPUM.data_path import dip_data

# Parameters
var_noise = [10 ** (-37), 10 ** (-2), 10 ** (-1)]
gamma = [10 ** (-35), 0.015, 1.5]

# Data
img_path = dip_data('original_DIP.tif')
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

M, N = f.shape

# Fourier transform
F = np.fft.fftshift(np.fft.fft2(f))

# Blur filter (centered)
H = motionBlurTF4e(M, N, 0.1, 0.1, 1)

# Filtering in the Fourier domain
G = F * H
g = np.real(np.fft.ifft2(G))

# Added noise (3 different variances)
z = np.zeros((M, N))
zn1 = np.random.normal(0.0, np.sqrt(var_noise[0]), size=(M, N))
zn2 = np.random.normal(0.0, np.sqrt(var_noise[1]), size=(M, N))
zn3 = np.random.normal(0.0, np.sqrt(var_noise[2]), size=(M, N))

Zn1 = np.fft.fft2(zn1)
Zn2 = np.fft.fft2(zn2)
Zn3 = np.fft.fft2(zn3)

Gn1 = G + Zn1
Gn2 = G + Zn2
Gn3 = G + Zn3

# Case 1. High noise, high Gamma
L = constrainedLsTF4e(H, gamma[2])
Fh = L * Gn3
fHatHighNoiseHighGamma = np.abs(np.real(np.fft.ifft2(Fh)))

# Case 2. Medium noise, Medium Gamma
L = constrainedLsTF4e(H, gamma[1])
Fh = L * Gn2
fHatMediumNoiseMediumGamma = np.abs(np.real(np.fft.ifft2(Fh)))

# Case 3. Low noise, low Gamma
L = constrainedLsTF4e(H, gamma[0])
Fh = L * Gn1
fHatLowNoiseLowGamma = np.abs(np.real(np.fft.ifft2(Fh)))

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(fHatHighNoiseHighGamma, cmap='gray')
axes[0].axis('off')

axes[1].imshow(fHatMediumNoiseMediumGamma, cmap='gray')
axes[1].axis('off')

axes[2].imshow(fHatLowNoiseLowGamma, cmap='gray')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure530.png')
plt.show()
