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
Order = "both"
NIter = 300

# Data
img_path = dip_data("noisy-elliptical-object.tif")
g = imread(img_path)

# Load initial snake data (overrides g if present, as in MATLAB)
mat = loadmat("Figure112.mat")

# Edge map
emap = snakeMap4e(g, T, Sig, NSig, Order)

# Snake force using plain gradient
FTx, FTy = snakeForce4e(emap, "gradient")

# Normalize forces
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)

# Threshold by magnitude to suppress small vectors
mag = np.sqrt(FTx**2 + FTy**2)
FTx = np.where(mag > 0.35, FTx, 0)
FTy = np.where(mag > 0.35, FTy, 0)

# Reduce density
FTxr = FTx[::15, ::15]
FTyr = FTy[::15, ::15]

# Display
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.quiver(np.flipud(FTyr), np.flipud(-FTxr), angles="xy", scale_units="xy", scale=1)
ax.set_title("Vector snake force")
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("Figure114.png")
plt.show()
