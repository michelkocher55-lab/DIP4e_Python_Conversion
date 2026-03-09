import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure95

# %% Data
# Keep PIL reader: this TIFF can be interpreted inverted by some loaders.
f = np.array(PIL.Image.open(dip_data("wirebond-mask.tif")))

# %% SE
B1 = ia.iasebox(5)
B2 = ia.iasebox(7)
B3 = ia.iasebox(22)

# %% Erosion
f1 = ia.iaero(f, B1)
f2 = ia.iaero(f, B2)
f3 = ia.iaero(f, B3)

# %% Display
fig = plt.figure(num=1, figsize=(10, 10))
try:
    fig.canvas.manager.set_window_title("Figure 9.5")
except Exception:
    pass

plt.subplot(2, 2, 1)
plt.imshow(f, cmap="gray")
plt.title("f")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(f1, cmap="gray")
plt.title(r"f1 = $\epsilon_{B1}(f)$")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(f2, cmap="gray")
plt.title(r"f2 = $\epsilon_{B2}(f)$")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(f3, cmap="gray")
plt.title(r"f3 = $\epsilon_{B3}(f)$")
plt.axis("off")

plt.tight_layout()
fig.savefig("Figure95.png", dpi=150, bbox_inches="tight")
plt.show()
