"""Figure 11.8 - Snake segmentation of 957-by-1024 rose."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from skimage.io import imread

from libDIPUM.snake_manual_input import snake_manual_input
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIPUM.snake_display import snake_display
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data


print("Running Figure118...")


def _im2double_like(a: np.ndarray) -> np.ndarray:
    """_im2double_like."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64)
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float64) / 65535.0
    if a.dtype == np.int16:
        return a.astype(np.float64) / 32768.0
    return a.astype(np.float64)


# Data
img_path = dip_data("rose957by1024.tif")
f = imread(img_path)
if f.ndim == 3:
    f = f[..., 0]

script_dir = os.path.dirname(__file__)
mat_path = os.path.join(script_dir, "Figure118.mat")

if os.path.exists(mat_path):
    data = loadmat(mat_path)
    if "f" in data:
        f = np.asarray(data["f"])
    xi = np.asarray(data["xi"]).squeeze()
    yi = np.asarray(data["yi"]).squeeze()
else:
    plt.figure()
    plt.imshow(f, cmap="gray")
    plt.axis("off")
    plt.title("Select initial snake")
    # Coordinates of initial snake selected manually.
    xi, yi = snake_manual_input(f, 150, "wo")
    savemat(mat_path, {"f": f, "xi": xi, "yi": yi})

# Parameters
T = 0.005
Sig = 11
NSig = 5
NIter = 400
Alpha = 10 * 0.05
Beta = 0.5
Gamma = 5

# Edge map
emap = snakeMap4e(f, T, Sig, NSig, "both")
emap = _im2double_like(intScaling4e(emap))

# Snake force
FTx, FTy = snakeForce4e(emap, "gradient")

# Normalize
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)

x = np.asarray(xi).copy()
y = np.asarray(yi).copy()

# Iterate
for _ in range(NIter):
    x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
    x, y = snakeReparam4e(x, y)

# Redistribute once more
x, y = snakeReparam4e(x, y)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].axis("off")
plt.sca(axes[0, 0])
snake_display(xi, yi, "g.")

axes[0, 1].imshow(emap, cmap="gray")
axes[0, 1].axis("off")

axes[1, 0].imshow(emap, cmap="gray")
axes[1, 0].axis("off")
plt.sca(axes[1, 0])
snake_display(x, y, "g.")

axes[1, 1].imshow(f, cmap="gray")
axes[1, 1].axis("off")
plt.sca(axes[1, 1])
snake_display(x, y, "g.")

out_path = os.path.join(script_dir, "Figure118.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
