import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from General.atmosphturb import atmosphturb
from libDIPUM.imnoise2 import imnoise2
from libDIP.constrainedLsTF4e import constrainedLsTF4e
from General.deconvreg1 import deconvreg1
from libDIPUM.data_path import dip_data

# Parameters
k = 0.0025
mu = 0
sigma = 10e-3
gamma = 5e-5

# Data
img_path = dip_data('aerial_view_no_turb.tif')
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

M, N = f.shape

# Fourier transform
F = np.fft.fftshift(np.fft.fft2(f))

# Atmospheric perturbations
H = atmosphturb(M, N, k)
h = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(H)))

# Filtering in the frequency domain
G = H * F
g = np.fft.ifft2(np.fft.fftshift(G))

# Add noise in the frequency domain
z = np.zeros((M, N))
zn, _ = imnoise2(z, 'gaussian', mu, sigma)
Gn = np.fft.fftshift(np.fft.fft2(zn))
G1 = G + Gn
g1 = np.fft.ifft2(np.fft.fftshift(G1))

# Restoration by using constrained least square (fixed gamma)
L = constrainedLsTF4e(H, gamma)
Fh = L * G1
fHat = np.fft.ifft2(Fh)

L1 = constrainedLsTF4e(H, gamma / 10.0)
Fh1 = L1 * G1
fHat1 = np.fft.ifft2(Fh1)

# Restoration by using constrained least square (regularized), iterative via fminbnd
fHat2, Lagra = deconvreg1(np.real(g1), np.real(h))

# Display
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
axes1[0, 0].imshow(np.abs(H), cmap='gray')
axes1[0, 0].set_title('Blurring transfer function')
axes1[0, 0].axis('off')

axes1[0, 1].imshow(np.abs(h), cmap='gray')
axes1[0, 1].set_title('Point spread function')
axes1[0, 1].axis('off')

axes1[1, 0].imshow(f, cmap='gray')
axes1[1, 0].set_title('Original image')
axes1[1, 0].axis('off')

axes1[1, 1].imshow(np.abs(g1), cmap='gray')
axes1[1, 1].set_title(f'Blurred + noise, sigma = {sigma}')
axes1[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure531.png')

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
axes2[0].imshow(np.abs(np.real(fHat)), cmap='gray')
axes2[0].set_title(f'Fixed : gamma = {gamma}')
axes2[0].axis('off')

axes2[1].imshow(np.abs(np.real(fHat1)), cmap='gray')
axes2[1].set_title(f'Fixed : gamma = {gamma/10.0}')
axes2[1].axis('off')

axes2[2].imshow(fHat2, cmap='gray')
axes2[2].set_title(f'Optimisation : gamma = {Lagra}')
axes2[2].axis('off')

plt.tight_layout()
plt.savefig('Figure531Bis.png')
plt.show()


