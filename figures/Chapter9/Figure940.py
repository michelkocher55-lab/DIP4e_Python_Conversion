import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import ia870 as ia
from General.AddSE2Image import AddSE2Image
from General.GPsnr import GPsnr
from General.matlab_hist import matlab_hist
from libDIPUM.data_path import dip_data

# %% Figure940
# Alternate Sequential Filtering

# %% Data
f = np.array(Image.open(dip_data("cygnusloop.tif")))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
Rad = [1, 3, 5]
B = [ia.iasedisk(r) for r in Rad]

# %% CO
CO = np.zeros((f.shape[0], f.shape[1], 3), dtype=f.dtype)
SNR_CO = np.zeros(3, dtype=float)
InfoCO = [""] * 3

for cpt in range(3):
    CO[:, :, cpt] = ia.iaclose(ia.iaopen(f, B[cpt]), B[cpt])
    gg = CO[:, :, cpt].astype(float)
    InfoCO[cpt] = f"C_{Rad[cpt]}(O_{Rad[cpt]}(f))"
    SNR_CO[cpt] = GPsnr(f.astype(float), gg)

# %% ASF (manual, then iaasf overwrite as in MATLAB script)
ASF = np.zeros((f.shape[0], f.shape[1], 3), dtype=f.dtype)
SNR_ASF = np.zeros(3, dtype=float)

for cpt in range(3):
    if cpt == 0:
        ASF[:, :, cpt] = ia.iaclose(ia.iaopen(f, B[cpt]), B[cpt])
    else:
        ASF[:, :, cpt] = ia.iaclose(ia.iaopen(ASF[:, :, cpt - 1], B[cpt]), B[cpt])
    hh = ASF[:, :, cpt].astype(float)
    SNR_ASF[cpt] = GPsnr(f.astype(float), hh)

for cpt in range(3):
    ASF[:, :, cpt] = ia.iaasf(f.astype(np.uint8), "CO", ia.iasecross(1), cpt + 1)
    gg = ASF[:, :, cpt].astype(float)
    SNR_ASF[cpt] = GPsnr(f.astype(float), gg)

InfoASF = [
    "C_1(O_1(f))",
    "C_3(O_3(C_1(O_1(f))))",
    "C_5(O_5(C_3(O_3(C_1(O_1(f))))))",
]

# %% Display figure 1 (CO)
fig1 = plt.figure(1, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(f, cmap="gray")
plt.title("f0")
plt.axis("off")

for iter_idx in range(3):
    plt.subplot(2, 2, 2 + iter_idx)
    mx = int(np.max(CO[:, :, iter_idx]))
    plt.imshow(
        AddSE2Image(CO[:, :, iter_idx], ia.iasedisk(Rad[iter_idx]), mx), cmap="gray"
    )
    plt.title(f"{InfoCO[iter_idx]}, SNR = {SNR_CO[iter_idx]:.3f} dB")
    plt.axis("off")

plt.tight_layout()
fig1.savefig("Figure940.png", dpi=150, bbox_inches="tight")

# %% Display figure 2 (ASF)
fig2 = plt.figure(2, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(f, cmap="gray")
plt.title("f0")
plt.axis("off")

for iter_idx in range(3):
    plt.subplot(2, 2, 2 + iter_idx)
    mx = int(np.max(ASF[:, :, iter_idx]))
    plt.imshow(
        AddSE2Image(ASF[:, :, iter_idx], ia.iasedisk(Rad[iter_idx]), mx), cmap="gray"
    )
    plt.title(f"{InfoASF[iter_idx]}, SNR = {SNR_ASF[iter_idx]:.3f} dB")
    plt.axis("off")

plt.tight_layout()
fig2.savefig("Figure940Bis.png", dpi=150, bbox_inches="tight")

# %% Display figure 3 (differences/hist)
fig3 = plt.figure(3, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(ia.iasubm(CO[:, :, -1], f), cmap="gray", vmin=0, vmax=50)
plt.title("f3 - f")
plt.axis("off")

plt.subplot(2, 2, 2)
e = f.astype(float) - CO[:, :, -1].astype(float)
emin, emax = float(np.min(e)), float(np.max(e))
centers = np.linspace(emin, emax, 10) if emax > emin else np.array([emin])
counts = matlab_hist(e.ravel(), centers)
plt.bar(centers, counts, width=(centers[1] - centers[0]) if centers.size > 1 else 1.0)
plt.title("Histogram of f - f3")

plt.subplot(2, 2, 3)
plt.imshow(ia.iasubm(ASF[:, :, -1], f), cmap="gray", vmin=0, vmax=50)
plt.title("ASF(f) - f")
plt.axis("off")

plt.subplot(2, 2, 4)
e1 = f.astype(float) - ASF[:, :, -1].astype(float)
e1min, e1max = float(np.min(e1)), float(np.max(e1))
centers1 = np.linspace(e1min, e1max, 10) if e1max > e1min else np.array([e1min])
counts1 = matlab_hist(e1.ravel(), centers1)
plt.bar(
    centers1, counts1, width=(centers1[1] - centers1[0]) if centers1.size > 1 else 1.0
)
plt.title("Histogram of f - ASF(f)")

plt.tight_layout()

plt.show()
