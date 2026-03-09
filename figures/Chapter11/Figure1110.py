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
NIter = 400

# Data
img_path = dip_data("U200.tif")
f = imread(img_path)

# Specify a circle
t = np.arange(0, 2 * np.pi + 0.1, 0.1)
xi = 100 + 80 * np.cos(t)
yi = 100 + 80 * np.sin(t)

# Close the snake
xi = np.concatenate([xi, [xi[0]]])
yi = np.concatenate([yi, [yi[0]]])

# Edge map
emap = snakeMap4e(f, 0.001, 3, 1, "after")

# Scale to range [0,1]
emap = intScaling4e(emap)

# Snake force (GVF)
FTx, FTy = snakeForce4e(emap, "gvf", 0.25, 80)

# Normalize
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + np.finfo(float).eps)
FTy = FTy / (mag + np.finfo(float).eps)

x = xi.copy()
y = yi.copy()

# Iterate
for _ in range(NIter):
    x, y = snakeIterate4e(0.05, 0.5, 5, x, y, 1, FTx, FTy)
    x, y = snakeReparam4e(x, y)

# Redistribute the points one last time
x, y = snakeReparam4e(x, y)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].axis("off")
axes[0, 0].plot(np.append(yi, yi[0]), np.append(xi, xi[0]), "g.")

axes[0, 1].imshow(emap, cmap="gray")
axes[0, 1].axis("off")

axes[1, 0].quiver(np.flipud(FTy[::2, ::2]), np.flipud(-FTx[::2, ::2]))
axes[1, 0].axis("off")

axes[1, 1].imshow(f, cmap="gray")
axes[1, 1].axis("off")
plt.sca(axes[1, 1])
snake_display(x, y, "g.")

plt.tight_layout()
plt.savefig("Figure1110.png")
plt.show()
