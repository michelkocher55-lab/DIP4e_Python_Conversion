import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.measure import find_contours
from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIPUM.data_path import dip_data

# Parameters
x0 = 250
y0 = 225
r = 120

a = -1
b = 1

iterations = [30, 160, 170, 226, 250]

# Data
img_path = dip_data("multiple-regions.tif")
f = img_as_float(imread(img_path))
M, N = f.shape

# Binarize
fbin = (f > 0.7).astype(float)

# Process
phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
contours_list = [find_contours(phi0, level=0)]

F = levelSetForce4e("binary", [fbin, a, b])

for niter in iterations:
    phi = phi0.copy()
    for _ in range(niter):
        phi = levelSetIterate4e(phi, F)
    contours_list.append(find_contours(phi, level=0))

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for idx in range(len(contours_list)):
    ax = axes.flat[idx]
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    for cont in contours_list[idx]:
        ax.plot(cont[:, 1], cont[:, 0], "g.")

plt.tight_layout()
plt.savefig("Figure1120.png")
plt.show()
