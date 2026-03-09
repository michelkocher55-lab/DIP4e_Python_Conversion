import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from General.atmosphturb import atmosphturb
from libDIP.lpFilterTF4e import lpFilterTF4e
from libDIPUM.imnoise2 import imnoise2
from General.deconvwnr import deconvwnr
from libDIPUM.data_path import dip_data

# Parameters
k = 0.0025
mu = 0
sigma = 1e-10
D0 = 85
order = 10

# Data
img_path = dip_data("aerial_view_no_turb.tif")
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

M, N = f.shape

# Fourier transform
F = np.fft.fftshift(np.fft.fft2(f))

# Atmospheric perturbations
H = atmosphturb(M, N, k)

# Filtering in the frequency domain
G = H * F
g = np.fft.ifft2(np.fft.fftshift(G))

# Add noise in the frequency domain
z = np.zeros((M, N))
zn, _ = imnoise2(z, "gaussian", mu, sigma)
Gn = np.fft.fftshift(np.fft.fft2(zn))
G1 = G + Gn
g1 = np.fft.ifft2(np.fft.fftshift(G1))

# Straight inverse filter
Fh1 = G1 / H
fHat = []
fHat.append(np.abs(np.real(np.fft.ifft2(Fh1))))

# Low-pass cutoff
LowPass = lpFilterTF4e("butterworth", M, N, D0, order)
Temp = LowPass * Fh1
fHat.append(np.abs(np.real(np.fft.ifft2(Temp))))

# Wiener
nspr = (sigma**2) / np.var(f)
psf = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(H)))
fHat.append(deconvwnr(g1, psf, nspr))

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    axes[i].imshow(fHat[i], cmap="gray")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("Figure528.png")
plt.show()
