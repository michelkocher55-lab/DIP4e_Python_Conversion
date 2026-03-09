import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path

# Add project root so local packages can be imported when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ia870 as ia
from helpers.AddSE2Image import AddSE2Image
from libDIPUM.data_path import dip_data

# %% Figure940New
# Alternate Sequential Filtering

# %% Data
f = np.array(Image.open(dip_data("cygnusloop.tif")))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
Rad = list(range(1, 6))
B = [ia.iasedisk(r) for r in Rad]

# %% CO
CO = np.zeros((f.shape[0], f.shape[1], len(Rad)), dtype=f.dtype)
InfoCO = [""] * len(Rad)
for cpt in range(len(Rad)):
    CO[:, :, cpt] = ia.iaclose(ia.iaopen(f, B[cpt]), B[cpt])
    InfoCO[cpt] = f"f_{cpt + 1} = C_{Rad[cpt]}(O_{Rad[cpt]}(f_{cpt}))"

# %% ASF
ASF = np.zeros((f.shape[0], f.shape[1], len(Rad)), dtype=f.dtype)
for cpt in range(len(Rad)):
    ASF[:, :, cpt] = ia.iaasf(f, "CO", B[cpt])

InfoASF = [
    "C_1(O_1(f))",
    "C_2(O_2(C_1(O_1(f))))",
    "C_3(O_3(C_2(O_2(C_1(O_1(f))))))",
    "ASFCO_4(f)",
    "ASFCO_5(f)",
]

# %% ASFRec
ASFRec = np.zeros((f.shape[0], f.shape[1], len(Rad)), dtype=f.dtype)
for cpt in range(len(Rad)):
    ASFRec[:, :, cpt] = ia.iaasfrec(
        f.astype(np.uint8), "CO", ia.iasecross(1), ia.iasecross(1), cpt + 1
    )

InfoASFRec = [
    "ASFRec(f, CO, 1)",
    "ASFRec(f, CO, 2)",
    "ASFRec(f, CO, 3)",
    "ASFRec(f, CO, 4)",
    "ASFRec(f, CO, 5)",
]

# %% Display figure 1 (CO)
fig1 = plt.figure(1, figsize=(12, 8))
try:
    fig1.canvas.manager.set_window_title("Figure 9.40")
except Exception:
    pass

plt.subplot(2, 3, 1)
plt.imshow(f, cmap="gray")
plt.title("f0")
plt.axis("off")

for iter_idx in range(len(Rad)):
    plt.subplot(2, 3, 2 + iter_idx)
    mx = int(np.max(CO[:, :, iter_idx]))
    plt.imshow(AddSE2Image(CO[:, :, iter_idx], B[iter_idx], mx), cmap="gray")
    plt.title(InfoCO[iter_idx])
    plt.axis("off")

plt.tight_layout()
fig1.savefig("Figure940New.png", dpi=150, bbox_inches="tight")

# %% Display figure 2 (ASF)
fig2 = plt.figure(2, figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(f, cmap="gray")
plt.title("f0")
plt.axis("off")

for iter_idx in range(len(Rad)):
    plt.subplot(2, 3, 2 + iter_idx)
    mx = int(np.max(ASF[:, :, iter_idx]))
    plt.imshow(AddSE2Image(ASF[:, :, iter_idx], B[iter_idx], mx), cmap="gray")
    plt.title(InfoASF[iter_idx])
    plt.axis("off")

plt.tight_layout()
fig2.savefig("Figure940NewBis.png", dpi=150, bbox_inches="tight")

# %% Display figure 3 (ASFRec)
fig3 = plt.figure(3, figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(f, cmap="gray")
plt.title("f0")
plt.axis("off")

for iter_idx in range(len(Rad)):
    plt.subplot(2, 3, 2 + iter_idx)
    mx = int(np.max(ASFRec[:, :, iter_idx]))
    plt.imshow(AddSE2Image(ASFRec[:, :, iter_idx], B[iter_idx], mx), cmap="gray")
    plt.title(InfoASFRec[iter_idx])
    plt.axis("off")

plt.tight_layout()
fig3.savefig("Figure940NewTer.png", dpi=150, bbox_inches="tight")

plt.show()
