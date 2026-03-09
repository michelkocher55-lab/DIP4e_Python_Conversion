import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure934
# Border clearing 9.5-31

# %% Init
Fig = 1

# %% Mask
img_name = dip_data("text-image.tif")
Mask = imread(img_name)
if Mask.ndim == 3:
    Mask = Mask[..., 0]
Mask = Mask > 0

# %% Marker
Marker = np.zeros_like(Mask, dtype=bool)
Marker[0, :] = True
Marker[-1, :] = True
Marker[:, 0] = True
Marker[:, -1] = True
Marker = np.logical_and(Marker, Mask)

# %% Reconstruction
BorderChar = ia.iainfrec(Marker, Mask, ia.iasecross(1))
NonBorderChar = ia.iasubm(Mask, BorderChar)

# %% Display
fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, num=Fig)
ax = ax.ravel()

ax[0].imshow(Mask, cmap="gray")
ax[0].set_title("Mask")
ax[0].axis("off")

ax[1].imshow(Marker, cmap="gray")
ax[1].set_title("Marker = Frame and Mask")
ax[1].axis("off")

ax[2].imshow(BorderChar, cmap="gray")
ax[2].set_title("Border characters from reconstruction")
ax[2].axis("off")

ax[3].imshow(NonBorderChar, cmap="gray")
ax[3].set_title("Non-border characters = Mask minus BorderChar")
ax[3].axis("off")

plt.tight_layout()
fig.savefig("Figure934.png", dpi=150, bbox_inches="tight")
plt.show()
