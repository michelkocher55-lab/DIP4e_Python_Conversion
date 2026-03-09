import matplotlib.pyplot as plt
from skimage.io import imread
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure97

# %% Data
# historical filename in MATLAB source: Fig0907(a)(text_gaps_1_and_2_pixels).tif
f = imread(dip_data("text-broken.tif"))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
B1 = ia.iasecross(1)

# %% Dilation
f1 = ia.iadil(f, B1)

# %% Display
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, num=1)
try:
    fig.canvas.manager.set_window_title("Figure 9.7")
except Exception:
    pass

ax[0].imshow(f, cmap="gray")
ax[0].set_title("f")
ax[0].axis("off")

ax[1].imshow(f1, cmap="gray")
ax[1].set_title(r"f1 = $\delta_{B1}(f)$")
ax[1].axis("off")

plt.tight_layout()
fig.savefig("Figure97.png", dpi=150, bbox_inches="tight")
plt.show()
