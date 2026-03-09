import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

from helpers.graycomatrix import graycomatrix
from helpers.graycoprops import graycoprops

print("Running Figure1232 (GLCM texture analysis)...")

Names = ["strip-uniform-noise", "strip-2Dsinusoidal-waveform", "strip-cktboard-section"]

f = []
for name in Names:
    img = imread(dip_data(f"{name}.tif"))
    if img.ndim == 3:
        img = img[:, :, 0]
    f.append(img)

# Co-occurrence matrix and properties
glcm = []
props = []
MaxProb = np.zeros(len(Names), dtype=float)
Entropy = np.zeros(len(Names), dtype=float)

for i in range(len(Names)):
    # Horizontal, one-pixel distance
    G, _ = graycomatrix(f[i], NumLevels=256, Offset=np.array([[0, 1]]))
    G2 = G[:, :, 0]
    glcm.append(G2)

    st = graycoprops(G2, ["contrast", "correlation", "homogeneity", "energy"])
    props.append(
        {
            "Contrast": float(st["Contrast"][0]),
            "Correlation": float(st["Correlation"][0]),
            "Homogeneity": float(st["Homogeneity"][0]),
            "Energy": float(st["Energy"][0]),
        }
    )

    p = G2 / np.sum(G2) if np.sum(G2) > 0 else G2.astype(float)
    MaxProb[i] = float(np.max(p)) if p.size else 0.0

    nz = p > 0
    Entropy[i] = -np.sum(p[nz] * np.log2(p[nz])) if np.any(nz) else 0.0

# Display
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
ax = ax.ravel()

for i in range(len(Names)):
    ax[i].imshow(f[i], cmap="gray")
    ax[i].set_title(
        f"p_max = {MaxProb[i]:.2g}, "
        f"rho = {props[i]['Correlation']:.2g}, "
        f"C = {props[i]['Contrast']:.3g}"
    )
    ax[i].axis("off")

    if i == 2:
        ax[i + 3].imshow(glcm[i], cmap="gray", vmin=0, vmax=40)
    else:
        ax[i + 3].imshow(glcm[i], cmap="gray")

    ax[i + 3].set_title(
        f"U = {props[i]['Energy']:.2g}, "
        f"H = {props[i]['Homogeneity']:.3g}, "
        f"E = {Entropy[i]:.2g}"
    )
    ax[i + 3].axis("off")

fig.tight_layout()
fig.savefig("Figure1232.png")

print("Saved Figure1232.png")
plt.show()
