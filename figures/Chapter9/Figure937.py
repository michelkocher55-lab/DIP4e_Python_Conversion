import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure937

# %% Data
f = np.array(Image.open(dip_data("circuitboard-section.tif")))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
B1 = ia.iasedisk(2)

# %% Erosion and dilation
f1 = ia.iaero(f, B1)
f2 = ia.iadil(f, B1)

# %% Display
fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, num=1)
try:
    fig.canvas.manager.set_window_title("Figure 9.35")
except Exception:
    pass

ax = ax.ravel()

ax[0].imshow(f, cmap="gray")
ax[0].set_title("f")
ax[0].axis("off")

ax[1].axis("off")

ax[2].imshow(f1, cmap="gray")
ax[2].set_title("f1 = erosion with disk radius 2")
ax[2].axis("off")

ax[3].imshow(f2, cmap="gray")
ax[3].set_title("f2 = dilation with disk radius 2")
ax[3].axis("off")

plt.tight_layout()
fig.savefig("Figure937.png", dpi=150, bbox_inches="tight")
plt.show()
