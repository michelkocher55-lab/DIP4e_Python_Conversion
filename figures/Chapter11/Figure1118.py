from typing import Any
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.measure import find_contours
from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIPUM.data_path import dip_data

# Parameters
x0 = [100, 100]
y0 = [350, 350]
r = [60, 150]
a = [1, 1]
b = [0, -1]
iterations = [100, 400, 900]

# Data
img_path = dip_data("letterA-distorted.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Binarize
fbin = (f < 0.6).astype(float)  # Letter should be white


# Helpers
def contourc_zero(phi: Any):
    """contourc_zero."""
    return find_contours(phi, level=0)


# Initial level set function located outside character.
phi0 = [None, None]
phi0[0] = levelSetFunction4e("circular", M, N, x0[0], y0[0], r[0])

c = [[], []]
c[0].append(contourc_zero(phi0[0]))

# Force definition for top row
F = [None, None]
F[0] = levelSetForce4e("binary", [fbin, a[0], b[0]])

# Show stages of curve evolution
for niter in iterations:
    phi = phi0[0].copy()
    for _ in range(niter):
        phi = levelSetIterate4e(phi, F[0])
    c[0].append(contourc_zero(phi))

# Initial level set function located inside character.
phi0[1] = levelSetFunction4e("circular", M, N, x0[1], y0[1], r[1])

c[1].append(contourc_zero(phi0[1]))

# Force definition for bottom row
F[1] = levelSetForce4e("binary", [fbin, a[1], b[1]])

# Show stages of curve evolution
for niter in iterations:
    phi = phi0[1].copy()
    for _ in range(niter):
        phi = levelSetIterate4e(phi, F[1])
    c[1].append(contourc_zero(phi))

# Display top row
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
for idx in range(len(iterations) + 1):
    ax = axes1.flat[idx]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    for cont in c[0][idx]:
        # cont is (row, col)
        ax.plot(cont[:, 1], cont[:, 0], "g.")

plt.tight_layout()
plt.savefig("Figure1118.png")

# Display bottom row
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
for idx in range(len(iterations) + 1):
    ax = axes2.flat[idx]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    for cont in c[1][idx]:
        ax.plot(cont[:, 1], cont[:, 0], "g.")

plt.tight_layout()
plt.savefig("Figure1118Bis.png")
plt.show()
