import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.measure import find_contours
import scipy.io as sio

from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e
from libDIPUM.curve_manual_input import curve_manual_input
from libDIPUM.coord2mask import coord2mask
from libDIPUM.curve_display import curve_display
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("noisy-blobs.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Input initial phi manually.
if os.path.exists("Figure1132.mat"):
    mat = sio.loadmat("Figure1132.mat")
    x = mat["y"].squeeze()
    y = mat["x"].squeeze()
else:
    x, y, vx, vy = curve_manual_input(f, 200, "g.")
    sio.savemat("Figure1132.mat", {"x": x, "y": y, "vx": vx, "vy": vy})

binmask = coord2mask(M, N, x, y)

# Parameters
mu_list = [0.5, 0.75, 3]
nu = 0
lambda1 = 1
lambda2 = 1
NIter = 1500

# Initial level set function.
phi0 = levelSetFunction4e("mask", binmask)

phi_list = []
c_list = []
for mu in mu_list:
    phi = phi0.copy()
    C = 0.5
    for i in range(1, NIter + 1):
        F = levelSetForce4e(
            "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
        )
        phi = levelSetIterate4e(phi, F)
        if i % 5 == 0:
            phi = levelSetReInit4e(phi, 5, 0.5)
    phi_list.append(phi)
    c_list.append(find_contours(phi, level=0.0))

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for idx in range(len(mu_list)):
    ax = axes[0, idx]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    for cc in c_list[idx]:
        # find_contours returns (row, col); curve_display(x,y) plots plot(y,x).
        curve_display(cc[:, 0], cc[:, 1], "g.", ax=ax)

    ax2 = axes[1, idx]
    ax2.imshow(phi_list[idx] <= 0, cmap="gray")
    ax2.axis("off")

plt.tight_layout()
plt.savefig("Figure1133.png")
plt.show()
