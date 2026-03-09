import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import scipy.io as sio

from libDIP.SnakeSegmentation import SnakeSegmentation
from libDIP.LevelSetEdgebased import LevelSetEdgebased
from libDIP.LevelSetRegionBased import LevelSetRegionBased
from libDIPUM.coord2mask import coord2mask
from libDIPUM.snake_display import snake_display
from libDIPUM.curve_display import curve_display
from libDIPUM.data_path import dip_data

# %% Data
img_path = dip_data("noisy-blobs.tif")
f = img_as_float(imread(img_path))
if f.ndim == 3:
    f = f[..., 0]
M, N = f.shape

# %% Initial boundary
mat = sio.loadmat("Figure1132.mat")
# Keep the same Figure1132.mat convention used in Chapter 11 scripts:
# stored x/y are swapped relative to the snake/level-set internal row/col convention.
x = np.asarray(mat["y"]).squeeze()
y = np.asarray(mat["x"]).squeeze()

# Binary mask
binmask = coord2mask(M, N, x, y)

# %% Parameters
# Snake
Snake = {
    "Mu": 0.2,
    "NIterConvergence": 180,
    "NIterForce": 160,
    "T": 0.005,
    "Alpha": 0.10,
    "Beta": 1.00,
    "Gamma": 0.20,
    "Sig": 11,
    "NSig": 5,
}

# Level set (edge based)
LSEdgeBased = {
    "HSize": 11,
    "Sigma": 3,
    "p": 1,
    "lambda": 50,
    "niter": 2000,
}

# Level set (region based)
LSRegionBased = {
    "mu": 0.5,
    "nu": 0,
    "lambda1": 1,
    "lambda2": 1,
    "niter": 2000,
}

# %% 1) Snake
x_snake, y_snake, emap_snake = SnakeSegmentation(
    f,
    x,
    y,
    Snake["T"],
    Snake["Sig"],
    Snake["NSig"],
    Snake["Mu"],
    Snake["NIterForce"],
    Snake["NIterConvergence"],
    Snake["Alpha"],
    Snake["Beta"],
    Snake["Gamma"],
)

# %% 2) Level set edge based
c0, fsmooth0, WBin0 = LevelSetEdgebased(
    f,
    binmask,
    LSEdgeBased["HSize"],
    LSEdgeBased["Sigma"],
    LSEdgeBased["p"],
    LSEdgeBased["lambda"],
    LSEdgeBased["niter"],
)

# %% 3) Level set region based
c = LevelSetRegionBased(
    f,
    binmask,
    LSRegionBased["mu"],
    LSRegionBased["nu"],
    LSRegionBased["lambda1"],
    LSRegionBased["lambda2"],
    LSRegionBased["niter"],
)

# %% Display
plt.figure(figsize=(13, 4))

plt.subplot(1, 3, 1)
plt.imshow(f, cmap="gray")
plt.axis("off")
snake_display(x_snake[0::2], y_snake[0::2], "yo")
plt.title("Snake enclosing the 3 blobs")

plt.subplot(1, 3, 2)
plt.imshow(f, cmap="gray")
plt.axis("off")
plt.title("Edge based level set")
curve_display(c0[1, :], c0[0, :], "y.")

plt.subplot(1, 3, 3)
plt.imshow(f, cmap="gray")
plt.axis("off")
plt.title("region based level set")
curve_display(c[1, :], c[0, :], "y.")

plt.tight_layout()
plt.savefig("Figure1137.png")
plt.show()
