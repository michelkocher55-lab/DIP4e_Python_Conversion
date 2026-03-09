from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu
import ia870 as ia
from helpers.mmshow import mmshow
from libDIPUM.data_path import dip_data

# %% Figure945

# %% Data
f = np.array(Image.open(dip_data("dark-blobs-on-light-background.tif")))
if f.ndim == 3:
    f = f[..., 0]

# %% Closing the dark patches
g = ia.iaclose(f, ia.iasedisk(29, "2D", "OCTAGON"))

# %% Closing the clear patches
g1 = ia.iaopen(g, ia.iasedisk(59, "2D", "OCTAGON"))

# %% Computing the boundary
level_val = threshold_otsu(g1)
level_norm = float(level_val) / 255.0
X = g1 > level_val
Y = ia.iagradm(X, ia.iasecross(1), ia.iasecross(0))
Yd = ia.iadil(Y, ia.iasecross(2))

# %% Display
try:
    _raw = input("Enonce (1) ou corrige (2) : ").strip()
    Choix = int(_raw) if _raw else 1
except Exception:
    Choix = 1

fig1 = plt.figure(1, figsize=(10, 8))
try:
    fig1.canvas.manager.set_window_title("Figure 9.45")
except Exception:
    pass

if Choix == 1:
    plt.subplot(1, 2, 1)
    plt.imshow(f, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    mmshow(f, Yd, Yd, Yd)
    plt.title("f with boundary Y")
    plt.axis("off")
else:
    ax1 = plt.subplot(2, 2, 1)
    mmshow(f, Yd, Yd, Yd)
    plt.title("f with boundary Y")
    plt.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(g, cmap="gray", vmin=0, vmax=255)
    plt.title("g = closing with disk radius 29")
    plt.axis("off")

    ax3 = plt.subplot(2, 2, 3)
    plt.imshow(g1, cmap="gray", vmin=0, vmax=255)
    plt.title("g1 = opening with disk radius 59")
    plt.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    plt.imshow(Yd, cmap="gray")
    plt.title(f"Boundary of thresholded g1, Otsu={int(round(level_norm * 255))}")
    plt.axis("off")

plt.tight_layout()

# Figure with histograms
fig2 = plt.figure(2, figsize=(10, 8))

bins = np.arange(0, 256)


def _hist255(img: Any):
    """_hist255."""
    c, _ = np.histogram(np.asarray(img).ravel(), bins=np.arange(-0.5, 256.5, 1.0))
    return c


plt.subplot(2, 2, 1)
plt.bar(bins, _hist255(f))
plt.title("hist(f)")

plt.subplot(2, 2, 2)
plt.bar(bins, _hist255(g))
plt.title("hist(g)")

plt.subplot(2, 2, 3)
plt.bar(bins, _hist255(g1))
plt.title("hist(g1)")

plt.tight_layout()

# %% Print
if Choix == 1:
    fig1.savefig("Figure945Enonce.png", dpi=150, bbox_inches="tight")
else:
    fig1.savefig("Figure945.png", dpi=150, bbox_inches="tight")

plt.show()
