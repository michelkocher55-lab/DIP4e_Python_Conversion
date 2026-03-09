import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from General.atmosphturb import atmosphturb
from libDIP.lpFilterTF4e import lpFilterTF4e
from libDIPUM.imnoise2 import imnoise2
from libDIPUM.data_path import dip_data

# Parameters
k = 0.0025
mu = 0
sigma = 10 ** (-10)
D0 = [40, 70, 85]
order = 10

# Data
img_path = dip_data("aerial_view_no_turb.tif")
f_orig = imread(img_path)
if f_orig.ndim == 3:
    f_orig = rgb2gray(f_orig)
f = img_as_float(f_orig)

M, N = f.shape

# Fourier transform (shifted)
F = np.fft.fftshift(np.fft.fft2(f))

# Atmospheric perturbations
H = atmosphturb(M, N, k)

# Filtering in the frequency domain
G = H * F

# Add noise in the frequency domain
z = np.zeros((M, N))
zn, _ = imnoise2(z, "gaussian", mu, sigma)
Gn = np.fft.fftshift(np.fft.fft2(zn))
G1 = G + Gn

# Straight inverse filter
Fh1 = G1 / H
fHat = [np.abs(np.real(np.fft.ifft2(Fh1)))]

# Cut off range of values of Fh1
Temp = []
for d0 in D0:
    LowPass = lpFilterTF4e("butterworth", M, N, d0, order)
    Temp.append(LowPass * Fh1)
    fHat.append(np.abs(np.real(np.fft.ifft2(Temp[-1]))))

# Display 1
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
axes1[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes1[0, 0].set_title("f")
axes1[0, 0].axis("off")

axes1[0, 1].imshow(np.log10(1 + np.abs(F)), cmap="gray")
axes1[0, 1].set_title("log(1+|F|)")
axes1[0, 1].axis("off")

axes1[0, 2].imshow(H, cmap="gray")
axes1[0, 2].set_title("|H|")
axes1[0, 2].axis("off")

axes1[1, 0].imshow(np.log10(1 + np.abs(G)), cmap="gray")
axes1[1, 0].set_title("log(1+|G|, G = H*F)")
axes1[1, 0].axis("off")

axes1[1, 1].imshow(np.log10(1 + np.abs(G1)), cmap="gray")
axes1[1, 1].set_title("log(1+|G1|, = G + G_n)")
axes1[1, 1].axis("off")

axes1[1, 2].axis("off")

plt.tight_layout()
plt.savefig("Figure527.png")

# Display 2
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
axes2[0, 0].imshow(np.log10(1 + np.abs(Fh1)), cmap="gray")
axes2[0, 0].set_title("log(1+|Fh1|, = Fh1 = G1 / H)")
axes2[0, 0].axis("off")

for i, d0 in enumerate(D0):
    ax = axes2.flat[i + 1]
    ax.imshow(np.log10(1 + np.abs(Temp[i])), cmap="gray")
    ax.set_title("log(1+|Fh1|, = Fh1 = (G1 / H) * LowPass)")
    ax.axis("off")

plt.tight_layout()
plt.savefig("Figure527Bis.png")

# Display 3
fig3, axes3 = plt.subplots(2, 2, figsize=(10, 10))
axes3[0, 0].imshow(fHat[0], cmap="gray")
axes3[0, 0].set_title("f_hat = ifft2 (Fh1)")
axes3[0, 0].axis("off")

for i, d0 in enumerate(D0):
    ax = axes3.flat[i + 1]
    ax.imshow(fHat[i + 1], cmap="gray")
    ax.set_title(f"f_hat = ifft2 (Fh1 * LowPass(D0, N)), D0 = {d0}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("Figure527Ter.png")
plt.show()
