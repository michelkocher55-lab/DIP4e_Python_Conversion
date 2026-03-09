import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e
from libDIPUM.contourc import contourc
from libDIPUM.curve_display import curve_display
from libDIPUM.data_path import dip_data

# Parameters
iterations = [500, 1000, 1500, 2000, 3500]
mu = 2.0
nu = 0.0
lambda1 = 1
lambda2 = 1

# Data
img_path = dip_data("cygnusloop.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Initial level set function.
x0 = int(round(M / 2))
y0 = int(round(N / 2))
r = 110
phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
c_list = [contourc(phi0, [0, 0])]

# Iterate
for niter in iterations:
    phi = phi0.copy()
    C = 0.5
    for i in range(1, niter + 1):
        F = levelSetForce4e(
            "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
        )
        phi = levelSetIterate4e(phi, F, 0.5)
        if i % 5 == 0:
            phi = levelSetReInit4e(phi, 5, 0.5)
    c_list.append(contourc(phi, [0, 0]))

# Display contours
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for idx in range(len(iterations) + 1):
    ax = axes.flat[idx]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    cc = c_list[idx]
    if cc.shape[1] > 0:
        curve_display(cc[1, 0::4], cc[0, 0::4], "r.", ax=ax)

plt.tight_layout()
plt.savefig("Figure1131.png")
plt.show()

# Binary mask display
plt.figure()
plt.imshow(phi <= 0, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.savefig("Figure1131Bis.png")
plt.show()
