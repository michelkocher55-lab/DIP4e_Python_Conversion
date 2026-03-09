import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import scipy.io as sio

from libDIP.SnakeSegmentation import SnakeSegmentation
from libDIP.LevelSetEdgebased import LevelSetEdgebased
from libDIP.LevelSetRegionBased import LevelSetRegionBased
from libDIPUM.snake_display import snake_display
from libDIPUM.curve_display import curve_display
from libDIPUM.data_path import dip_data

# %% Parameters
# Snake
Snake = {
    "T": 0.01,
    "Sig": 11,
    "NSig": 5,
    "Mu": 0.2,
    "NIterForce": 160,
    "NIterConvergence": 35,
    "Alpha": 0.05,
    "Beta": 0.5,
    "Gamma": 2.5,
}

# Level set (edge based)
LSEdgeBased = {
    "HSize": 21,
    "Sigma": 5,
    "p": 1,
    "lambda": 50,
    "niter": 500,
}

# Level set (region based)
LSRegionBased = {
    "mu": 2.0,
    "nu": 0.0,
    "lambda1": 1,
    "lambda2": 1,
    "niter": 3500,
}

# %% Data
img_path = dip_data("cygnusloop.tif")
f = img_as_float(imread(img_path))
if f.ndim == 3:
    f = f[..., 0]
M, N = f.shape
_ = (M, N)

# %% Initial contour / mask
mat_snake = sio.loadmat("WorkspaceFig1138(a).mat")
xi = np.asarray(mat_snake["xi"]).squeeze()
yi = np.asarray(mat_snake["yi"]).squeeze()

mat_mask = sio.loadmat("WorkspaceForFig1138(b).mat")
mask = np.asarray(mat_mask["mask"])

# %% 1) Snake
x, y, emap = SnakeSegmentation(
    f,
    xi,
    yi,
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
    mask,
    LSEdgeBased["HSize"],
    LSEdgeBased["Sigma"],
    LSEdgeBased["p"],
    LSEdgeBased["lambda"],
    LSEdgeBased["niter"],
)

# %% 3) Level set region based
c = LevelSetRegionBased(
    f,
    mask,
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
snake_display(xi[0::2], yi[0::2], "r.")
snake_display(x[0::2], y[0::2], "y.")
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
plt.savefig("Figure1138.png")
plt.show()
