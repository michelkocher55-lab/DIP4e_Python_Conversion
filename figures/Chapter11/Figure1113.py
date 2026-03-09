from typing import Any
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIPUM.snake_display import snake_display
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Parameters
NIter = [10, 20, 40, 60, 80]
T = 0.01
Sig = 11
NSig = 5

# Data
img_path = dip_data("breast-implant.tif")
if len(sys.argv) > 1:
    img_path = sys.argv[1]

f = imread(img_path)

# Edge map
emap = snakeMap4e(f, T, Sig, NSig, "both")
# Scale to [0, 1] range
emap = intScaling4e(emap)

# Initial contour
t = np.arange(0, 2 * np.pi + 0.05, 0.05)
xi = 300 + 70 * np.cos(t)
yi = 360 + 90 * np.sin(t)

# Close the snake
xi = np.concatenate([xi, [xi[0]]])
yi = np.concatenate([yi, [yi[0]]])

# Snake force
FTx, FTy = snakeForce4e(emap, "gvf", 0.2, 160)

# Normalize
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)


def script_for_fig1113(xi: Any, yi: Any, FTx: Any, FTy: Any, n_iter: Any):
    """script_for_fig1113."""
    x = xi.copy()
    y = yi.copy()
    for _ in range(n_iter):
        x, y = snakeIterate4e(0.05, 0.5, 2.5, x, y, 1, FTx, FTy)
        x, y = snakeReparam4e(x, y)
    x, y = snakeReparam4e(x, y)
    return x, y


# Process
xs = []
ys = []
for n_iter in NIter:
    x, y = script_for_fig1113(xi, yi, FTx, FTy, n_iter)
    xs.append(x)
    ys.append(y)

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].axis("off")
plt.sca(axes[0, 0])
snake_display(xi[::2], yi[::2], "g.")

for idx, n_iter in enumerate(NIter):
    ax = axes.flat[idx + 1]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    plt.sca(ax)
    snake_display(xs[idx][::2], ys[idx][::2], "g.")

plt.tight_layout()
plt.savefig("Figure1113.png")
plt.show()
