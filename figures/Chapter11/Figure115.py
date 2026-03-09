import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.io import loadmat
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIPUM.data_path import dip_data

# Parameters
T = 0.001
Sig = 15
NSig = 3
Order = "after"

# Data
img_path = dip_data("noisy-elliptical-object.tif")
g = imread(img_path)

# Load initial snake data (overrides g if present)
mat = loadmat("Figure112.mat")

# Edge map
emap = snakeMap4e(g, T, Sig, NSig, Order)

# Snake force using plain gradient
FTx, FTy = snakeForce4e(emap, "gradient")

# Normalize the forces
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)

# Reduce density
FTxr = FTx[::10, ::10]
FTyr = FTy[::10, ::10]

# Display
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.quiver(np.flipud(FTyr), np.flipud(-FTxr), angles="xy", scale_units="xy", scale=1)
ax.set_title("Vector snake force")
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("Figure115.png")
plt.show()
