import numpy as np
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
iterations = [200, 300, 400, 600, 800]

# Data
img_path = dip_data('rose479by512.tif')
f = img_as_float(imread(img_path))
M, N = f.shape

# Circular initial phi
x0 = int(round(M / 2))
y0 = int(round(N / 2))
r = max(x0, y0) - max(x0, y0) / 5.0
phi0 = levelSetFunction4e('circular', M, N, x0, y0, r)
c_list = [contourc(phi0, [0, 0])]

# Smooth image (none for this figure)
fsmooth = f

# Edge-marking function
W = levelSetForce4e('gradient', [fsmooth, 1, 50])

# Threshold W
T = (np.max(W) + np.min(W)) / 2.0
W = W > T

# Process
for niter in iterations:
    phi = phi0.copy()
    C = 0.5
    for i in range(1, niter + 1):
        F = levelSetForce4e('geodesic', [phi, C, W])
        phi = levelSetIterate4e(phi, F)
        if i % 5 == 0:
            phi = levelSetReInit4e(phi, 5, 0.5)
    c_list.append(contourc(phi, [0, 0]))

# Display
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for idx in range(len(iterations) + 1):
    ax = axes.flat[idx]
    ax.imshow(f, cmap='gray')
    ax.axis('off')
    cc = c_list[idx]
    curve_display(cc[1, :], cc[0, :], 'r.', ax=ax)

plt.tight_layout()
plt.savefig('Figure1129.png')
plt.show()
