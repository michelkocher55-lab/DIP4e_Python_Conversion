from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.fftpack import dct
from libDIP.tmat4e import tmat4e
from libDIPUM.lpfilter import lpfilter
from libDIPUM.data_path import dip_data


def dct2(a: Any):
    """dct2."""
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


# Parameters
D0 = 60

# Data
f = img_as_float(imread(dip_data("characterTestPattern688.tif")))
M, N = f.shape
P = 2 * M
Q = 2 * N

# Filter design in the Fourier domain
H_noncenter = lpfilter("ideal", M, N, D0).astype(float)
H_center = np.fft.fftshift(H_noncenter)

# Fourier transform
F_Fourier = np.fft.fft2(f)

# By using matrix multiplication
A_DFT = tmat4e("DFT", M)
F1_Fourier = M * A_DFT @ f @ A_DFT.T
Temp = (1 / M) * A_DFT.conj().T @ F1_Fourier @ A_DFT.conj()

# Cosine transform
F_DCT = dct2(f)
A_DCT = tmat4e("DCT", M)
F1_DCT = A_DCT @ f @ A_DCT.T

# Sine transform
A_DST = tmat4e("DST", M)
F1_DST = A_DST @ f @ A_DST.T

# Hartley transform
A_DHT = tmat4e("DHT", M)
F1_DHT = A_DHT @ f @ A_DHT.T

# Filtering in the Fourier domain
G_Fourier = np.fft.fftshift(F_Fourier) * H_center
g_Fourier = np.real(np.fft.ifft2(np.fft.ifftshift(G_Fourier)))

# Filtering in the Hartley domain
G_Hartley = F1_DHT * H_noncenter
g_Hartley = A_DHT.conj().T @ G_Hartley @ A_DHT.conj()

# Filtering in the Cosine domain
G_Cosine = F1_DCT * H_noncenter
g_Cosine = A_DCT.conj().T @ G_Cosine @ A_DCT.conj()

# Filtering in the Sine domain
G_Sine = F1_DST * H_noncenter
g_Sine = A_DST.conj().T @ G_Sine @ A_DST.conj()

# Display
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(np.log10(np.abs(np.fft.fftshift(F_Fourier))), cmap="gray")
axes[0, 1].set_title("|F.Fourier|")
axes[0, 1].axis("off")

axes[0, 2].imshow(g_Fourier, cmap="gray")
axes[0, 2].set_title("g_Fourier")
axes[0, 2].axis("off")

axes[1, 0].imshow(np.log10(np.abs(np.fft.fftshift(F1_DHT))), cmap="gray")
axes[1, 0].set_title("|F.Hartley|")
axes[1, 0].axis("off")

axes[1, 1].imshow(np.log10(np.abs(F1_DCT)), cmap="gray")
axes[1, 1].set_title("|F.Cosine|")
axes[1, 1].axis("off")

axes[1, 2].imshow(np.log10(np.abs(F1_DST)), cmap="gray")
axes[1, 2].set_title("|F.Sine|")
axes[1, 2].axis("off")

axes[2, 0].imshow(g_Hartley, cmap="gray")
axes[2, 0].set_title("g_Hartley")
axes[2, 0].axis("off")

axes[2, 1].imshow(g_Cosine, cmap="gray")
axes[2, 1].set_title("g_Cosine")
axes[2, 1].axis("off")

axes[2, 2].imshow(g_Sine, cmap="gray")
axes[2, 2].set_title("g_Sine")
axes[2, 2].axis("off")

plt.tight_layout()
plt.savefig("Figure615.png")
plt.show()
