"""Figure 11.7 - Different starting configuration for segmenting ellipse."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIPUM.snake_display import snake_display
from libDIPUM.data_path import dip_data


print("Running Figure117...")


def Process(alpha, beta, gamma, x, y, FTx, FTy, n_iter):
    # Iterate.
    for _ in range(n_iter):
        x, y = snakeIterate4e(alpha, beta, gamma, x, y, 1, FTx, FTy)
        x, y = snakeReparam4e(x, y)

    # Redistribute one last time.
    x, y = snakeReparam4e(x, y)
    return x, y


# Parameters
Alpha = 0.05
Beta = [0.0, 2.0, 4.0]
Gamma = 0.6
T = 0.001
Sig = 15
NSig = 3
NIter = [200, 400, 400]

# Data
img_path = dip_data('noisy-elliptical-object.tif')
g = imread(img_path)

# Initial snake coordinates: circle
# MATLAB: t = 0:0.05:2*pi
t = np.arange(0.0, 2.0 * np.pi + 0.05, 0.05)
xi = 320.0 + 200.0 * np.cos(t)
yi = 320.0 + 200.0 * np.sin(t)

# Close snake
xi = np.concatenate([xi, [xi[0]]])
yi = np.concatenate([yi, [yi[0]]])

# Edge map and external forces
emap = snakeMap4e(g, T, Sig, NSig, "both")
FTx, FTy = snakeForce4e(emap, "gradient")

# Normalize force
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)

# Process for each beta
x_res = []
y_res = []
for k in range(len(Beta)):
    xk, yk = Process(Alpha, Beta[k], Gamma, xi.copy(), yi.copy(), FTx, FTy, NIter[k])
    x_res.append(xk)
    y_res.append(yk)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(g, cmap="gray")
axes[0, 0].axis("off")
axes[0, 0].plot(np.r_[yi, yi[0]], np.r_[xi, xi[0]], ".g")

axes[0, 1].imshow(g, cmap="gray")
axes[0, 1].axis("off")
plt.sca(axes[0, 1])
snake_display(x_res[0], y_res[0], ".g")

axes[1, 0].imshow(g, cmap="gray")
axes[1, 0].axis("off")
plt.sca(axes[1, 0])
snake_display(x_res[1], y_res[1], ".g")

axes[1, 1].imshow(g, cmap="gray")
axes[1, 1].axis("off")
plt.sca(axes[1, 1])
snake_display(x_res[2], y_res[2], ".g")

out_path = os.path.join(os.path.dirname(__file__), "Figure117.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
