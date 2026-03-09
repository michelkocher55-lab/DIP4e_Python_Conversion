import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.intScaling4e import intScaling4e
from libDIP.lpFilterTF4e import lpFilterTF4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("characterTestPattern688.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Compute fft
P = 2 * M
Q = 2 * N
F = np.fft.fft2(f, s=(P, Q))
Power = np.abs(F) ** 2
Power_shift = np.fft.fftshift(Power)
PT = np.sum(Power)

# Energy enclosed by radii
radii = [10, 30, 60, 160, 460]
E = []
for k in radii:
    H = lpFilterTF4e("ideal", P, Q, k)
    prod = H * Power_shift
    E.append(np.sum(prod) / PT)
E = np.array(E)

# Build circle mask
Y, X = np.indices((P, Q))
cy = P // 2
cx = Q // 2
C = np.zeros((P, Q), dtype=bool)
for k in radii:
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    C |= np.abs(dist - k) <= 1  # thickness ~2

# Display spectrum with circles
S = np.fft.fftshift(np.log(1 + np.abs(F)))
S = intScaling4e(S)
S[C] = 1

# Reduce size 50%
S = S[::2, ::2]

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(f, cmap="gray")
axes[0].axis("off")

energy_str = np.array2string(E, precision=2, floatmode="fixed")
axes[1].imshow(S, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Energy = {energy_str}")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure440.png")
plt.show()
