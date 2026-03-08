import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.recnotch import recnotch
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data('satellite_original.tif')
f_orig = imread(img_name)
if f_orig.ndim == 3: f_orig = f_orig[:,:,0]
f = img_as_float(f_orig)

M, N = f.shape

# DFT
F = np.fft.fft2(f)

S = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(F))))

# Notch filter design
HR = recnotch('reject', 'vertical', M, N, 19, 30)

# Filtering in the Fourier domain
G = F * HR
g = np.real(np.fft.ifft2(G))

# Obtain interference pattern using a bandpass filter
HP = 1 - HR
GP = F * HP
gp = intScaling4e(np.real(np.fft.ifft2(GP)))

# Spectrum times notch filter
SF = S * np.fft.fftshift(HR)

# Display (5 plots in Figure518)
fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(f, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(S, cmap='gray')
ax2.set_title('Spectrum of the original')
ax2.axis('off')

ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(np.fft.fftshift(HR), cmap='gray')
ax3.set_title('Notch filter')
ax3.axis('off')

ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(SF, cmap='gray')
ax4.set_title('Spectrum * Notch')
ax4.axis('off')

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(g, cmap='gray')
ax5.set_title('Recovered')
ax5.axis('off')

# Leave subplot (2,3,6) empty intentionally
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

plt.tight_layout()
plt.savefig('Figure518.png')
print("Saved Figure518.png")

# Separate figure: interference
plt.figure(figsize=(6, 6))
plt.imshow(gp, cmap='gray')
plt.title('Interference')
plt.axis('off')
plt.tight_layout()
plt.savefig('Figure519.png')
print("Saved Figure519.png")

plt.show()
