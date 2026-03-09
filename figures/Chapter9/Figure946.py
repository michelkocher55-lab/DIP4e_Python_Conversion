import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu
import ia870 as ia
from General.AddSE2Image import AddSE2Image
from libDIPUM.data_path import dip_data

# %% Figure946
# Segment calculator symbols

# %% Parameters
Bc4 = ia.iasecross()

# %% Data
Mask = np.array(Image.open(dip_data("calculator.tif")))
if Mask.ndim == 3:
    Mask = Mask[..., 0]

# %% Thresholding
Level = threshold_otsu(Mask)
X = Mask > Level

# %% Structuring elements
B1 = ia.iaseline(71, 0)
B2 = ia.iaseline(11, 0)
B3 = ia.iaseline(21, 0)

# %% Removing clear horizontal objects (Morphological)
Marker = ia.iaero(Mask, B1)
g1 = ia.iadil(Marker, B1)
OK = np.all(g1 == ia.iaopen(Mask, B1))
print(f"OK = {OK}")

# %% Removing clear horizontal objects (Geodesical)
g2 = ia.iainfrec(Marker, Mask, Bc4)
Mask1 = ia.iasubm(Mask, g2)

# %% Removing clear vertical objects (Geodesical)
Marker1 = ia.iaero(Mask1, B2)
g4 = ia.iainfrec(Marker1, Mask1, Bc4)
g5 = ia.iadil(g4, B3)
Marker2 = ia.iaintersec(Mask1, g5)
g6 = ia.iainfrec(Marker2, Mask1, Bc4)

# %% Final thresholding
Level1 = threshold_otsu(g6)
X1 = g6 > Level1

# %% Display figure 1
fig1, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1)
try:
    fig1.canvas.manager.set_window_title("Figure 9.46")
except Exception:
    pass

ax = ax.ravel()
ax[0].imshow(Mask, cmap="gray")
ax[0].set_title("Mask")
ax[0].axis("off")

ax[1].imshow(X, cmap="gray")
ax[1].set_title(f"X threshold on Mask, Otsu={int(round(Level))}")
ax[1].axis("off")

ax[2].imshow(AddSE2Image(Marker, B1, int(np.max(Marker))), cmap="gray")
ax[2].set_title("Marker = erosion with line 71")
ax[2].axis("off")

ax[3].imshow(g1, cmap="gray")
ax[3].set_title("g1 = opening with line 71")
ax[3].axis("off")

ax[4].imshow(g2, cmap="gray")
ax[4].set_title("g2 = reconstruction from Marker in Mask")
ax[4].axis("off")

ax[5].imshow(Mask1, cmap="gray")
ax[5].set_title("Mask1 = Mask minus g2")
ax[5].axis("off")

fig1.tight_layout()
fig1.savefig("Figure946.png", dpi=150, bbox_inches="tight")

# %% Display figure 2
fig2, bx = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, num=2)
try:
    fig2.canvas.manager.set_window_title("Figure 9.46 bis")
except Exception:
    pass

bx = bx.ravel()
bx[0].imshow(Mask1, cmap="gray")
bx[0].set_title("Mask1 = Mask minus g2")
bx[0].axis("off")

bx[1].imshow(Marker1, cmap="gray")
bx[1].set_title("Marker1 = erosion with line 11")
bx[1].axis("off")

bx[2].imshow(g4, cmap="gray")
bx[2].set_title("g4 = reconstruction from Marker1 in Mask1")
bx[2].axis("off")

bx[3].imshow(g5, cmap="gray")
bx[3].set_title("g5 = dilation of g4 with line 21")
bx[3].axis("off")

bx[4].imshow(Marker2, cmap="gray")
bx[4].set_title("Marker2 = Mask1 and g5")
bx[4].axis("off")

bx[5].imshow(g6, cmap="gray")
bx[5].set_title("g6 = reconstruction from Marker2 in Mask1")
bx[5].axis("off")

fig2.tight_layout()
fig2.savefig("Figure946Bis.png", dpi=150, bbox_inches="tight")

# %% Display figure 3
fig3, cx = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, num=3)
try:
    fig3.canvas.manager.set_window_title("Figure 9.46 ter")
except Exception:
    pass

cx = cx.ravel()

cx[0].imshow(Mask, cmap="gray")
cx[0].set_title("Mask")
cx[0].axis("off")

cx[1].imshow(X, cmap="gray")
cx[1].set_title(f"X threshold on Mask, Otsu={int(round(Level))}")
cx[1].axis("off")

cx[2].imshow(X1, cmap="gray")
cx[2].set_title(f"X1 threshold on g6, Otsu={int(round(Level1))}")
cx[2].axis("off")

cx[3].axis("off")

fig3.tight_layout()
fig3.savefig("Figure946Ter.png", dpi=150, bbox_inches="tight")

plt.show()
