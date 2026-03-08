import os
import numpy as np
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

# Parameters
mu = 0.5
nu = 0
lambda1 = 1
lambda2 = 1
iterations = [500, 1000, 1500]

# Data
img_path = dip_data('noisy-blobs.tif')
f = img_as_float(imread(img_path))
M, N = f.shape

# Input initial phi manually.
if os.path.exists('Figure1132.mat'):
    mat = sio.loadmat('Figure1132.mat')
    x = mat['y'].squeeze()
    y = mat['x'].squeeze()
else:
    x, y, vx, vy = curve_manual_input(f, 200, 'g.')
    sio.savemat('Figure1132.mat', {'x': x, 'y': y, 'vx': vx, 'vy': vy})

binmask = coord2mask(M, N, x, y)

# Create and display initial level set function.
phi_list = [levelSetFunction4e('mask', binmask)]
c_list = [find_contours(phi_list[0], level=0.0)]

for niter in iterations:
    phi = phi_list[0].copy()
    C = 0.5
    for i in range(1, niter + 1):
        F = levelSetForce4e('regioncurve', [f, phi, mu, nu, lambda1, lambda2], ['Fn', 'Cn'])
        phi = levelSetIterate4e(phi, F)
        if i % 5 == 0:
            phi = levelSetReInit4e(phi, 5, 0.5)
    phi_list.append(phi)
    c_list.append(find_contours(phi, level=0.0))

# Display
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx in range(len(iterations) + 1):
    ax = axes[0, idx]
    ax.imshow(f, cmap='gray')
    ax.axis('off')
    for cc in c_list[idx]:
        # find_contours gives (row, col). curve_display(x,y) does plot(y,x),
        # so pass x=row, y=col to display as plot(col,row).
        curve_display(cc[:, 0], cc[:, 1], 'g.', ax=ax)

    ax2 = axes[1, idx]
    ax2.imshow(phi_list[idx] <= 0, cmap='gray')
    ax2.axis('off')

plt.tight_layout()
plt.savefig('Figure1132.png')
plt.show()
