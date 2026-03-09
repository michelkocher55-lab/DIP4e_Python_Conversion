import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.io import loadmat, savemat
from libDIPUM.snake_manual_input import snake_manual_input
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIPUM.snake_display import snake_display
from libDIPUM.data_path import dip_data

# Parameters
Alpha = 0.5
Beta = 2.0
Gamma = 5.0
NPoints = 150
T = 0.001
Sig = 15
NSig = 3
Order = "both"
NIter = 300

# Data
img_path = dip_data("noisy-elliptical-object.tif")
g = imread(img_path)

# Display image and get initial snake manually.
if os.path.exists("Figure112.mat"):
    mat = loadmat("Figure112.mat")
    xi = mat["xi"].squeeze()
    yi = mat["yi"].squeeze()
else:
    xi, yi = snake_manual_input(g, NPoints, "go")
    savemat("Figure112.mat", {"xi": xi, "yi": yi})

# Close the snake
xi = np.concatenate([xi, [xi[0]]])
yi = np.concatenate([yi, [yi[0]]])

# Edge map using filtering before and after edge map is computed.
emap = snakeMap4e(g, T, Sig, NSig, Order)

# Snake force using plain gradient.
FTx, FTy = snakeForce4e(emap, "gradient")

# Normalize the forces.
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)

x = xi.copy()
y = yi.copy()

# Iterate
for _ in range(NIter):
    x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
    x, y = snakeReparam4e(x, y)

# Display figure 1
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
axes1[0, 0].imshow(g, cmap="gray")
axes1[0, 0].set_title("Original image")
axes1[0, 0].axis("off")

axes1[0, 1].imshow(emap, cmap="gray")
axes1[0, 1].set_title("edge map")
axes1[0, 1].axis("off")

axes1[1, 0].imshow(FTx, cmap="gray")
axes1[1, 0].set_title("FT_x")
axes1[1, 0].axis("off")

axes1[1, 1].imshow(FTy, cmap="gray")
axes1[1, 1].set_title("FT_y")
axes1[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure112.png")

# Display figure 2
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
r = 10
ax2.quiver(np.flipud(FTy[::r, ::r]), np.flipud(-FTx[::r, ::r]))
ax2.set_title("Vector snake force")
plt.tight_layout()
plt.savefig("Figure112Bis.png")

# Display figure 3
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
ax3.imshow(g, cmap="gray")
ax3.plot(xi, yi, "or")
ax3.set_title("Initial contour")
ax3.axis("off")
plt.tight_layout()
plt.savefig("Figure112Ter.png")

# Display figure 4
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))
ax4.imshow(g, cmap="gray")
snake_display(x, y, "go")
plt.tight_layout()
plt.savefig("Figure112Quart.png")

plt.show()
