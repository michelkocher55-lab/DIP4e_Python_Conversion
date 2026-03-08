import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

iterations = [30, 250]

# Data
img_path = dip_data('multiple-regions.tif')
f = img_as_float(imread(img_path))
M, N = f.shape

# Binarize
fbin = (f > 0.7).astype(float)

# Process
phi0 = levelSetFunction4e('circular', M, N, x0, y0, r)
contours_list = [find_contours(phi0, level=0)]

F = levelSetForce4e('binary', [fbin, a, b])

phi_list = []
for niter in iterations:
    phi = phi0.copy()
    for _ in range(niter):
        phi = levelSetIterate4e(phi, F)
    phi_list.append(phi)
    contours_list.append(find_contours(phi, level=0))

# Display
fig = plt.figure(figsize=(12, 8))

# Top row: contours on image
for i in range(3):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(f, cmap='gray')
    ax.axis('off')
    for cont in contours_list[i]:
        ax.plot(cont[:, 1], cont[:, 0], 'g.')

# Bottom row: surface plots
for i in range(3):
    ax = fig.add_subplot(2, 3, i + 4, projection='3d')
    if i == 0:
        phi = phi0
    else:
        phi = phi_list[i - 1]
    phi_sub = phi[::8, ::8]
    X, Y = np.meshgrid(np.arange(phi_sub.shape[1]), np.arange(phi_sub.shape[0]))
    ax.plot_surface(X, Y, phi_sub, facecolor=(0.9, 0.9, 0.9), edgecolor='black', linewidth=0.3, shade=False, alpha=1.0)

    plane = np.zeros((round(N / 8), round(M / 8)))
    Xp, Yp = np.meshgrid(np.arange(plane.shape[1]), np.arange(plane.shape[0]))
    ax.plot_surface(Xp, Yp, plane, facecolor=(0.5, 0.5, 0.5), edgecolor='none', alpha=1.0)

    ax.view_init(elev=10, azim=232)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_axis_off()

plt.tight_layout()
plt.savefig('Figure1121.png')
plt.show()
