import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.paddedsize import paddedsize
from libDIPUM.dftuv import dftuv
from libDIPUM.lpfilter import lpfilter
from libDIPUM.data_path import dip_data

# Parameters
GammaH = 1.2
GammaL = 0.1
D0 = 5
MaxDisp = 0.1

# Data
img_path = dip_data("tun.jpg")
f = img_as_float(imread(img_path))
if f.ndim == 3:
    f = f[:, :, 0]
NR, NC = f.shape

# Padding (post zeros)
PQ = paddedsize(f.shape)
pad_rows = PQ[0] - NR
pad_cols = PQ[1] - NC
fp = np.pad(f, ((0, pad_rows), (0, pad_cols)), mode="constant")

# Filter design in the frequency domain
U, V = dftuv(PQ[0], PQ[1])
D = np.hypot(U, V)
HLP = lpfilter("gaussian", PQ[0], PQ[1], D0)
HHP = 1 - HLP
H = (GammaH - GammaL) * HHP + GammaL
Hc = np.fft.fftshift(H)

# Filtering in the frequency domain
fl = np.log(1 + fp)
F = np.fft.fft2(fl)
G = H * F
gp = np.fft.ifft2(G)

gp = np.real(np.exp(gp) - 1)

# Crop to original size
g = gp[:NR, :NC]

# Display figure 1
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
axes1[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1)
axes1[0, 0].set_title("f")
axes1[0, 0].axis("off")

vmin = np.min(g)
axes1[0, 1].imshow(g, cmap="gray", vmin=vmin, vmax=MaxDisp)
axes1[0, 1].set_title(f"g = Homorphic filter (f), Dyn = [{vmin:.6g}, {MaxDisp}]")
axes1[0, 1].axis("off")

axes1[1, 0].hist(f.ravel(), bins=256)
axes1[1, 0].set_title("Hist(f)")

axes1[1, 1].hist(g.ravel(), bins=256)
axes1[1, 1].set_title("Hist(g)")
axes1[1, 1].axvline(MaxDisp, color="black")

plt.tight_layout()

# Display figure 2
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))

logF = np.log10(np.abs(F))
finite = np.isfinite(logF)
Min = np.min(logF[finite])
Max = np.max(logF[finite])

cmap = plt.cm.gray.copy()
cmap.set_bad(color="black")

axes2[0, 0].imshow(np.fft.fftshift(logF), cmap=cmap, vmin=Min, vmax=Max)
axes2[0, 0].set_title("|F(u, v)|")
axes2[0, 0].axis("off")

axes2[0, 1].imshow(Hc, cmap="gray")
axes2[0, 1].set_title("H(u, v)")
axes2[0, 1].axis("off")

logG = np.log10(np.abs(G))
axes2[1, 0].imshow(np.fft.fftshift(logG), cmap=cmap, vmin=Min, vmax=Max)
axes2[1, 0].set_title("|F(u, v)*H(u, v)|")
axes2[1, 0].axis("off")

axes2[1, 1].plot(Hc[Hc.shape[0] // 2, :])
axes2[1, 1].axis("tight")

plt.tight_layout()

fig1.savefig("Figure460Tunnel_1.png")
fig2.savefig("Figure460Tunnel_2.png")
plt.show()
