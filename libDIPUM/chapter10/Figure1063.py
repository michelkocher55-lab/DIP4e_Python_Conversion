from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import ia870 as ia
from skimage.io import imread
from libDIPUM.data_path import dip_data
# import MKRLib

# Data
image_path = dip_data("corneacells.tif")
a = imread(image_path)
b = ia.iagsurf(a)

# Filtering and cell detection
c = ia.iaasf(a, "oc", ia.iasecross(), 2)
d = ia.iaregmax(c)

# Background marker
e = ia.ianeg(a)
f = ia.iacwatershed(e, d, ia.iasebox())

# Labeling markers and gradient
g = ia.iagray(f, "uint16", 1)
h1 = ia.iaaddm(ia.ialabel(d), 1)
h = ia.iaintersec(ia.iagray(d, "uint16"), h1)
i = ia.iaunion(g, h)

# Gradient
j = ia.iagradm(a)

# Constrained watershed from markers
k = ia.iacwatershed(j, i)


def _imshow_ready(x: Any):
    """Convert ia870 color output from (C,H,W) to (H,W,C) for matplotlib."""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
        x = np.moveaxis(x, 0, -1)
    return x


def _colorize_i_from_g_h(g_img: Any, h_img: Any, seed: Any = 0):
    """Create RGB display: red where g==1, random flat color for each label in h."""
    g_arr = np.asarray(g_img)
    h_arr = np.asarray(h_img)

    rgb = np.zeros(h_arr.shape + (3,), dtype=np.float32)

    labels = np.unique(h_arr)
    labels = labels[labels != 0]  # keep background as black
    rng = np.random.default_rng(seed)
    lut = {lab: rng.random(3, dtype=np.float32) for lab in labels}

    for lab in labels:
        rgb[h_arr == lab] = lut[lab]

    # Force red on pixels marked in g (priority over label colors).
    rgb[g_arr == 1] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return rgb


# Figure 1
fig1 = plt.figure(figsize=(10, 8))
ax = fig1.add_subplot(2, 2, 1)
ax.imshow(a, cmap="gray")
ax.set_title("a = Original image")
ax.axis("off")

ax = fig1.add_subplot(2, 2, 2)
ax.imshow(b, cmap="gray")
ax.set_title("Surf(a)")
ax.axis("off")

ax = fig1.add_subplot(2, 2, 3)
ax.imshow(c, cmap="gray")
ax.set_title("c = OC(a)")
ax.axis("off")

ax = fig1.add_subplot(2, 2, 4)
ax.imshow(d, cmap="gray")
ax.set_title("d = RMAX(c)")
ax.axis("off")

fig1.tight_layout()
fig1.savefig("Figure1063.png")

# Figure 2
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(2, 2, 1)
ax.imshow(ia.iagsurf(c), cmap="gray")
ax.set_title("Surf(c)")
ax.axis("off")

ax = fig2.add_subplot(2, 2, 2)
plt.sca(ax)
ax.imshow(_imshow_ready(ia.iagshow(ia.iagsurf(c), d)))
ax.set_title("Surf(c), d")
ax.axis("off")

ax = fig2.add_subplot(2, 2, 3)
plt.sca(ax)
ax.imshow(_imshow_ready(ia.iagshow(e, d)))
ax.set_title("e=~a, d")
ax.axis("off")

ax = fig2.add_subplot(2, 2, 4)
ax.imshow(f, cmap="gray")
ax.set_title("f = WS(e, d)")
ax.axis("off")

fig2.tight_layout()
fig2.savefig("Figure1063Bis.png")

# Figure 3
fig3 = plt.figure(figsize=(10, 8))
ax = fig3.add_subplot(2, 2, 1)
plt.sca(ax)
ax.imshow(_imshow_ready(ia.iagshow(e, f, d)))
ax.set_title("e, f, d")
ax.axis("off")

ax = fig3.add_subplot(2, 2, 2)
plt.sca(ax)
ax.imshow(_colorize_i_from_g_h(g, h), interpolation="nearest")
ax.set_title("i =")
ax.axis("off")

ax = fig3.add_subplot(2, 2, 3)
ax.imshow(j, cmap="gray")
ax.set_title("j = Grad(a)")
ax.axis("off")

ax = fig3.add_subplot(2, 2, 4)
plt.sca(ax)
ax.imshow(_imshow_ready(ia.iagshow(a, k, k)))
ax.set_title("a, k")
ax.axis("off")

fig3.tight_layout()
fig3.savefig("Figure1063Ter.png")

print("Saved Figure1063.png, Figure1063Bis.png, Figure1063Ter.png")
plt.show()
