import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIP.binaryRegionProps4e import binaryRegionProps4e

print("Running Figure1222 (Shape descriptors)...")

# Parameters
FileNames = [
    "wingding-circle-solid.tif",
    "wingding-star-6pt.tif",
    "wingding-square-solid.tif",
    "wingding-teardrop.tif",
]

base_dir = "/Users/michelkocher/michel/Data/DIP-DIPUM/DIP"

# Data
X = []
for name in FileNames:
    path = os.path.join(base_dir, name)
    img = imread(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    X.append(img > 0)

# Properties
Compactness1 = []
Circularity1 = []
Eccentricity1 = []
for x in X:
    p = binaryRegionProps4e(x)
    Compactness1.append(p["comp"])
    Circularity1.append(p["circ"])
    Eccentricity1.append(p["ecc"])

Compactness1 = np.array(Compactness1, dtype=float)
Circularity1 = np.array(Circularity1, dtype=float)
Eccentricity1 = np.array(Eccentricity1, dtype=float)

# Display 1
fig1, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.ravel()
for i in range(len(FileNames)):
    ax[i].imshow(X[i], cmap="gray", interpolation="nearest")
    ax[i].set_title(
        f"{Compactness1[i]:.6g}, {Circularity1[i]:.6g}, {Eccentricity1[i]:.6g}"
    )
    ax[i].axis("off")
fig1.tight_layout()
fig1.savefig("Figure1222.png")

# Display 2
fig2 = plt.figure(figsize=(10, 7))
ax3 = fig2.add_subplot(111, projection="3d")

for i, name in enumerate(FileNames):
    ax3.stem(
        [Circularity1[i]],
        [Eccentricity1[i]],
        [Compactness1[i]],
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
    )
    ax3.text(Circularity1[i], Eccentricity1[i], Compactness1[i], name)

ax3.set_xlabel("Circularity")
ax3.set_ylabel("Eccentricity")
ax3.set_zlabel("Compactness")
ax3.legend(["circle", "star", "square", "drop"])
ax3.grid(True)
ax3.view_init(elev=28, azim=134)

fig2.tight_layout()
fig2.savefig("Figure1223.png")

print("Saved Figure1222.png and Figure1223.png")
plt.show()
