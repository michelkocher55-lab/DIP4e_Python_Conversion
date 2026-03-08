import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.measure import find_contours
from scipy.io import loadmat, savemat
from scipy.ndimage import convolve
from libDIPUM.curve_manual_input import curve_manual_input
from libDIPUM.coord2mask import coord2mask
from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e
from libDIPUM.gaussKernel4e import gaussKernel4e
from libDIPUM.data_path import dip_data

# Parameters
n = 21
sig = 5
iterations = [200, 400, 600, 1000, 2000]

# Data
img_path = dip_data('breast-implant.tif')
f = img_as_float(imread(img_path))
M, N = f.shape

# Input initial phi manually
if os.path.exists('Figure1124.mat'):
    mat = loadmat('Figure1124.mat')
    x = mat['y'].squeeze() #MKR
    y = mat['x'].squeeze() #MKR
else:
    x, y, vx, vy = curve_manual_input(f, 200, 'g.')
    savemat('Figure1124.mat', {'x': x, 'y': y})

# Create mask for generating initial level set function
binmask = coord2mask(M, N, x, y)

# Create initial level set function
phi0 = levelSetFunction4e('mask', binmask)

# Obtain zero-level set contour
contours_list = [find_contours(phi0, level=0)]

# Smooth image
G = gaussKernel4e(n, sig)
fsmooth = convolve(f, G, mode='nearest')

# Compute edge-marking function
W = levelSetForce4e('gradient', [fsmooth, 1, 50])
T = (np.max(W) + np.min(W)) / 2.0
WBin = W > T

# Iterate for specified iterations
for niter in iterations:
    phi = phi0.copy()
    C = 0.5
    for i in range(1, niter + 1):
        F = levelSetForce4e('geodesic', [phi, C, WBin])
        phi = levelSetIterate4e(phi, F)
        if i % 5 == 0:
            phi = levelSetReInit4e(phi, 5, 0.5)
    contours_list.append(find_contours(phi, level=0))

# Save final phi
phi1500 = phi

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for idx in range(len(contours_list)):
    ax = axes.flat[idx]
    ax.imshow(f, cmap='gray')
    ax.axis('off')
    for cont in contours_list[idx]:
        ax.plot(cont[:, 1], cont[:, 0], 'y.')

plt.tight_layout()
plt.savefig('Figure1124.png')
plt.show()
