import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e
from libDIPUM.contourc import contourc
from libDIPUM.curve_display import curve_display
from libDIPUM.data_path import dip_data

# Parameters
mu = 0.5
nu = 0
lambda1 = 1
lambda2 = 1
iterations = [100, 300, 500, 700, 1000]

# Data
img_path = dip_data("rose479by512.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Circular initial phi
y, x = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
center = (int(round(M / 2)), int(round(N / 2)))
r = max(center) - max(center) / 3.0
phi_list = [np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) - r]

# Initial contour
c_list = [contourc(phi_list[0], [0, 0])]

for niter in iterations:
    phi = phi_list[0].copy()
    C = 0.5
    for i in range(1, niter + 1):
        F = levelSetForce4e(
            "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
        )
        phi = levelSetIterate4e(phi, F)
        if i % 5 == 0:
            phi = levelSetReInit4e(phi, 5, 0.5)
    phi_list.append(phi)
    c_list.append(contourc(phi, [0, 0]))

# Display
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for idx in range(len(iterations) + 1):
    ax = axes.flat[idx]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    cc = c_list[idx]
    curve_display(cc[1, :], cc[0, :], "g.", ax=ax)

plt.tight_layout()
plt.savefig("Figure1134.png")
plt.show()
