import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from skimage.measure import regionprops
import sys
from pathlib import Path
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure920

# %% Init
Fig = 1

# %% Parameters
Bc = ia.iasecross(1)

# %% Data
f = np.array(Image.open(dip_data("Chickenfilet-with-bones.tif")))
if f.ndim == 3:
    f = f[..., 0]

# %% Thresholding
Threshold = 200
X = f > Threshold

Bin = np.arange(0, 256)
Count, _ = np.histogram(f.ravel().astype(float), bins=np.arange(-0.5, 256.5, 1.0))

# %% Opening
B = ia.iasebox(1)
X1 = ia.iaopen(X, B)

# %% Labeling
L = ia.ialabel(X1, Bc)
L = np.asarray(L)

# %% Area stat
Stat = regionprops(L.astype(np.int32))
areas = [s.area for s in Stat]

# %% Display
fig = plt.figure(Fig, figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(f, cmap="gray")
plt.title("f")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.bar(Bin[1:], Count[1:])
plt.title("Hist(f)")
plt.axvline(Threshold, color="r")
plt.axis("tight")
plt.gca().set_box_aspect(1)

plt.subplot(2, 3, 3)
plt.imshow(X, cmap="gray")
plt.title(f"X = T_{{{int(round(Threshold))}}}(f)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(X1, cmap="gray")
plt.title(r"X1 = $\gamma_B(X)$")
plt.axis("off")

plt.subplot(2, 3, 5)
max_label = int(np.max(L))
if max_label > 0:
    lut = plt.cm.jet(np.linspace(0, 1, max_label + 1))
    lut[0, :3] = 0.0
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(lut[:, :3])
    plt.imshow(L, cmap=cmap, vmin=0, vmax=max_label)
else:
    plt.imshow(L, cmap="gray")
plt.title("L = Label_4(X1)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.bar(np.arange(1, len(areas) + 1), areas)
plt.xlabel("Connected Part")
plt.title("Area")
plt.gca().set_box_aspect(1)
plt.axis("tight")

plt.tight_layout()
fig.savefig("Figure920.png", dpi=150, bbox_inches="tight")
plt.show()
