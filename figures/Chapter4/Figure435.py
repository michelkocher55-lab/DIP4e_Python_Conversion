
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIPUM.lpfilter import lpfilter
from libDIPUM.dftfilt import dftfilt
from libDIPUM.paddedsize import paddedsize
from libDIPUM.data_path import dip_data

print("Running Figure435 (Gaussian Lowpass Filtering Steps)...")

# Parameters
D0 = 25

# Data Loading
img_path = dip_data('blown_ic_crop.tif')

f = imread(img_path)
if f.ndim == 3: f = f[:,:,0]
f = img_as_float(f)
M, N = f.shape

# Padding
# PQ = paddedsize(size(f))
PQ = paddedsize(f.shape)
# fp = padarray(f, [PQ(1)-M, PQ(2)-N], 'post') - defaults to zero padding
pad_h = PQ[0] - M
pad_w = PQ[1] - N
fp = np.pad(f, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

# Fourier Transform of padded image
# FP = fft2(fp)
FP = np.fft.fft2(fp)

# Filter Generation
# H = lpfilter('gaussian', PQ(1), PQ(2), D0)
# lpfilter returns centered or uncentered?
# My implementation of lpfilter uses dftuv which uses uncentered coordinates (0 to M-1 wrapped).
# So lpfilter returns UNCENTERED H (corners are DC).
H = lpfilter('gaussian', PQ[0], PQ[1], D0)

# Filtering
# G = H .* FP
G = H * FP

# gp = real(ifft2(G))
gp = np.real(np.fft.ifft2(G))

# Crop
# g = gp(1:M, 1:N)
g = gp[:M, :N]

# Compare with dftfilt
# g1 = dftfilt(f, H, 'symmetric')
# Wait, MATLAB script says `dftfilt(f, H, 'symmetric')`.
# `dftfilt` in MATLAB takes H. H is generated with PQ size (padded size).
# `dftfilt` logic: checks if H is 2M/2N or similar?
# Actually dftfilt pads f to size(H).
# H is PQ (padded size). f is M,N.
# dftfilt will pad f to PQ using 'symmetric' (reflection).
# But wait, step `fp` above used Zero padding ('post').
# So `g` comes from Zero padded input.
# `g1` comes from Symmetric padded input.
# So `g` and `g1` will differ at boundaries.
# I will replicate this exactly.

g1 = dftfilt(f, H, 'symmetric') # 'symmetric' -> mode='reflect' in my implementation

# Display
# Preparing spectra for display (Log magnitude, shifted)
FP_shifted = np.fft.fftshift(FP)
S_FP = np.log(1 + np.abs(FP_shifted))

H_shifted = np.fft.fftshift(H)

G_shifted = np.fft.fftshift(G)
S_G = np.log(1 + np.abs(G_shifted))

fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))

axes1[0, 0].imshow(f, cmap='gray')
axes1[0, 0].set_title(f'f, Size = {f.shape}')
axes1[0, 0].axis('off')

axes1[0, 1].imshow(fp, cmap='gray')
axes1[0, 1].set_title(f'fp (padded), Size = {fp.shape}')
axes1[0, 1].axis('off')

axes1[1, 0].imshow(S_FP, cmap='gray')
axes1[1, 0].set_title(f'DFT(fp), Size = {FP.shape}')
axes1[1, 0].axis('off')

axes1[1, 1].imshow(H_shifted, cmap='gray')
axes1[1, 1].set_title(f'H (Gaussian), Size = {H.shape}, D0={D0}')
axes1[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure435_1.png')

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

axes2[0, 0].imshow(S_G, cmap='gray')
axes2[0, 0].set_title(f'DFT(fp) * H, Size = {G.shape}')
axes2[0, 0].axis('off')

axes2[0, 1].imshow(gp, cmap='gray')
axes2[0, 1].set_title(f'gp (padded result), Size = {gp.shape}')
axes2[0, 1].axis('off')

axes2[1, 0].imshow(g, cmap='gray')
axes2[1, 0].set_title(f'g (cropped), Size = {g.shape}')
axes2[1, 0].axis('off')

axes2[1, 1].imshow(g1, cmap='gray')
axes2[1, 1].set_title('g using dftfilt (symmetric pad)')
axes2[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure435_2.png')

print("Saved Figure435_1.png and Figure435_2.png")
plt.show()
