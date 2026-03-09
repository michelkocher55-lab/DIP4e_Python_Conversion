import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

from helpers.graycomatrix import graycomatrix
from helpers.graycoprops import graycoprops

print("Running Figure1233 (Correlation vs horizontal offset)...")

# Data
Names = ["strip-uniform-noise", "strip-2Dsinusoidal-waveform", "strip-cktboard-section"]

f = []
for name in Names:
    img = imread(dip_data(f"{name}.tif"))
    if img.ndim == 3:
        img = img[:, :, 0]
    f.append(img)

# Co-occurrence matrix property (correlation) vs offset
Properties = np.zeros((len(Names), 50), dtype=float)

for cpt in range(len(Names)):
    for offset in range(1, 51):
        G, _ = graycomatrix(f[cpt], NumLevels=256, Offset=np.array([[0, offset]]))
        st = graycoprops(G, "correlation")
        Properties[cpt, offset - 1] = st["Correlation"][0]

# Display
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
ax = ax.ravel()

for i in range(len(Names)):
    ax[i].imshow(f[i], cmap="gray")
    ax[i].axis("off")

    ax[i + 3].plot(np.arange(1, 51), Properties[i, :])
    ax[i + 3].set_xlabel("Horizontal offset")
    ax[i + 3].set_title("correlation")
    ax[i + 3].set_xlim(1, 50)
    ax[i + 3].set_ylim(-1, 1)

fig.tight_layout()
fig.savefig("Figure1233.png")

print("Saved Figure1233.png")
plt.show()
