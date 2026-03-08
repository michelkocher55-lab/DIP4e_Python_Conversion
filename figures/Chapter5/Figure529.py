
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float, random_noise
from libDIP.motionBlurTF4e import motionBlurTF4e
from libDIP.pWienerTF4e import pWienerTF4e
from libDIPUM.data_path import dip_data

# Parameters
VarNoise = [1e-37, 1e-2, 1e-1]
VarWiener = [1e-35, 0.15e-1, 0.2e-1]

# Data
img_name = dip_data('original_DIP.tif')
f_orig = imread(img_name)
if f_orig.ndim == 3: f_orig = f_orig[:,:,0]
f = img_as_float(f_orig)

M, N = f.shape
F = np.fft.fft2(f)

# Blur filter
H_centered = motionBlurTF4e(M, N, 0.1, 0.1, 1)
H = np.fft.fftshift(H_centered)

# Filtering in Fourier domain
G = F * H
g = np.real(np.fft.ifft2(G))

# Added noise
z = np.zeros((M, N))

# Generate noise with clip=True to match MATLAB imnoise behavior on zeros
zn1 = random_noise(z, mode='gaussian', mean=0, var=VarNoise[0], clip=True)
zn2 = random_noise(z, mode='gaussian', mean=0, var=VarNoise[1], clip=True)
zn3 = random_noise(z, mode='gaussian', mean=0, var=VarNoise[2], clip=True)

Zn1 = np.fft.fft2(zn1)
Zn2 = np.fft.fft2(zn2)
Zn3 = np.fft.fft2(zn3)

Gn1 = G + Zn1
Gn2 = G + Zn2
Gn3 = G + Zn3

gn1 = np.real(np.fft.ifft2(Gn1))
gn2 = np.real(np.fft.ifft2(Gn2))
gn3 = np.real(np.fft.ifft2(Gn3))

# Performs filtering
# Inverse using pWienerTF4e(H, 0)
W0 = pWienerTF4e(H, 0)

fHatInvFilter1 = np.real(np.fft.ifft2(W0 * Gn1))
fHatInvFilter2 = np.real(np.fft.ifft2(W0 * Gn2))
fHatInvFilter3 = np.real(np.fft.ifft2(W0 * Gn3))

# Wiener
W1 = pWienerTF4e(H, VarWiener[0])
W2 = pWienerTF4e(H, VarWiener[1])
W3 = pWienerTF4e(H, VarWiener[2])

fHatWienerFilter1 = np.real(np.fft.ifft2(W1 * Gn1))
fHatWienerFilter2 = np.real(np.fft.ifft2(W2 * Gn2))
fHatWienerFilter3 = np.real(np.fft.ifft2(W3 * Gn3))

# Display
# Use vmin=0, vmax=1 to match MATLAB imshow(double) behavior

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
axes1[0].imshow(f, cmap='gray', vmin=0, vmax=1)
axes1[0].set_title('Original')
axes1[0].axis('off')
axes1[1].imshow(g, cmap='gray', vmin=0, vmax=1)
axes1[1].set_title('Blurred')
axes1[1].axis('off')
plt.tight_layout()
plt.savefig('Figure529_init.png')

fig2, axes2 = plt.subplots(3, 3, figsize=(12, 12))

# Row 1 High
axes2[0, 0].imshow(gn3, cmap='gray', vmin=0, vmax=1)
axes2[0, 0].set_title('Blurred + Noise (High)')
axes2[0, 0].axis('off')
axes2[0, 1].imshow(fHatInvFilter3, cmap='gray', vmin=0, vmax=1)
axes2[0, 1].set_title('Inverse Filter')
axes2[0, 1].axis('off')
axes2[0, 2].imshow(fHatWienerFilter3, cmap='gray', vmin=0, vmax=1)
axes2[0, 2].set_title('Wiener Filter')
axes2[0, 2].axis('off')

# Row 2 Med
axes2[1, 0].imshow(gn2, cmap='gray', vmin=0, vmax=1)
axes2[1, 0].set_title('Blurred + Noise (Med)')
axes2[1, 0].axis('off')
axes2[1, 1].imshow(fHatInvFilter2, cmap='gray', vmin=0, vmax=1)
axes2[1, 1].set_title('Inverse Filter')
axes2[1, 1].axis('off')
axes2[1, 2].imshow(fHatWienerFilter2, cmap='gray', vmin=0, vmax=1)
axes2[1, 2].set_title('Wiener Filter')
axes2[1, 2].axis('off')

# Row 3 Low
axes2[2, 0].imshow(gn1, cmap='gray', vmin=0, vmax=1)
axes2[2, 0].set_title('Blurred + Noise (Low)')
axes2[2, 0].axis('off')
axes2[2, 1].imshow(fHatInvFilter1, cmap='gray', vmin=0, vmax=1)
axes2[2, 1].set_title('Inverse Filter')
axes2[2, 1].axis('off')
axes2[2, 2].imshow(fHatWienerFilter1, cmap='gray', vmin=0, vmax=1)
axes2[2, 2].set_title('Wiener Filter')
axes2[2, 2].axis('off')

plt.tight_layout()
plt.savefig('Figure529.png')
print("Saved Figure529_init.png and Figure529.png")
plt.show()