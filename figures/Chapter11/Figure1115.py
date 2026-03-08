import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.measure import find_contours
from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIPUM.data_path import dip_data

# Parameters
r = 1
times = [50, 100, 300, 700, 800]

# Data
img_path = dip_data('gray-lakes.tif')
lakes = img_as_float(imread(img_path))
M, N = lakes.shape

# Initial level set function
x0 = int(round(M / 2))
y0 = int(round(N / 2))
phi = [levelSetFunction4e('circular', M, N, x0, y0, r)]

# Force field
f = (lakes > 0.9).astype(float)
F = levelSetForce4e('binary', [f, 1, 0])

# Process
contours = [None]  # 1-based indexing style
K = len(times)
for i in range(K):
    niter = times[i]
    phi_next = phi[0].copy()
    for _ in range(niter):
        phi_next = levelSetIterate4e(phi_next, F)
    phi.append(phi_next)
    # contourc(phi, [0 0]) equivalent: contours at level 0
    contours.append(find_contours(phi_next, level=0))

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for idx in range(K + 1):
    ax = axes.flat[idx]
    ax.imshow(lakes, cmap='gray')
    ax.axis('off')
    # Plot contours (skip index 0 since it is placeholder)
    if idx > 0:
        for c in contours[idx]:
            # c is (row, col); plot x=col, y=row
            ax.plot(c[:, 1], c[:, 0], 'g.')

plt.tight_layout()
plt.savefig('Figure1115.png')
plt.show()
