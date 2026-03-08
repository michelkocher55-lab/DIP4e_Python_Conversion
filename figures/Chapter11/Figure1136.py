import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.io import imread
from skimage.util import img_as_float
import scipy.io as sio

from libDIP.intScaling4e import intScaling4e
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIP.levelSetFunction4e import levelSetFunction4e
from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e

from libDIPUM.gaussKernel4e import gaussKernel4e
from libDIPUM.coord2mask import coord2mask
from libDIPUM.contourc import contourc
from libDIPUM.curve_display import curve_display
from libDIPUM.curve_manual_input import curve_manual_input
from libDIPUM.snake_display import snake_display
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('breast-implant.tif')
f = img_as_float(imread(img_path))
if f.ndim == 3:
    f = f[..., 0]
M, N = f.shape

# (1) Snake part (gvf)
T = 0.005
Sig = 11
NSig = 5
Alpha = 10 * 0.05
Beta = 0.5
Gamma = 5

t = np.arange(0, 2 * np.pi + 1e-12, 0.05)
xi = 300 + 70 * np.cos(t)
yi = 360 + 90 * np.sin(t)

# Close the snake
xi = np.append(xi, xi[0])
yi = np.append(yi, yi[0])

# Edge map
emap = snakeMap4e(f, T, Sig, NSig, 'both')
emap = img_as_float(intScaling4e(emap))

# Snake force
FTx, FTy = snakeForce4e(emap, 'gvf', 0.2, 160)
mag = np.sqrt(FTx**2 + FTy**2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)

# Process
x = xi.copy()
y = yi.copy()
for _ in range(80):
    x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
    x, y = snakeReparam4e(x, y)
xSnake, ySnake = snakeReparam4e(x, y)

# (2) Level Set (Edge based)
n = 21
sig = 5
NIter = 1500

if os.path.exists('Figure1124.mat'):
    mat = sio.loadmat('Figure1124.mat')
    # Keep same convention as Figure1124.py in this project
    x_ls = np.asarray(mat['y']).squeeze()
    y_ls = np.asarray(mat['x']).squeeze()
else:
    x_ls, y_ls, vx, vy = curve_manual_input(f, 200, 'g.')
    sio.savemat('Figure1124.mat', {'x': x_ls, 'y': y_ls, 'vx': vx, 'vy': vy})

binmask = coord2mask(M, N, x_ls, y_ls)
phi0 = levelSetFunction4e('mask', binmask)

G = gaussKernel4e(n, sig)
fsmooth = convolve(f, G, mode='nearest')

W = levelSetForce4e('gradient', [fsmooth, 1, 50])
T = (np.max(W) + np.min(W)) / 2.0
WBin = W > T

phi = phi0.copy()
C = 0.5
for I in range(1, NIter + 1):
    F = levelSetForce4e('geodesic', [phi, C, WBin])
    phi = levelSetIterate4e(phi, F)
    if I % 5 == 0:
        phi = levelSetReInit4e(phi, 5, 0.5)
cLevelSetEdge = contourc(phi, [0, 0])

# (3) Level Set (Region based) Chan-Vese
mu = 0.5
nu = 0
lambda1 = 1
lambda2 = 1

y1, x1 = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
center = [330, 350]
r = 30
phi0 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2) - r

phi = phi0.copy()
for I in range(1, 800 + 1):
    F = levelSetForce4e('regioncurve', [f, phi, mu, nu, lambda1, lambda2], ['Fn', 'Cn'])
    phi = levelSetIterate4e(phi, F)
    if I % 5 == 0:
        phi = levelSetReInit4e(phi, 5, 0.5)
cLevelSetRegion = contourc(phi, [0, 0])

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(f, cmap='gray')
plt.axis('off')
snake_display(xSnake, ySnake, 'g.')
plt.title('snake')

plt.subplot(1, 3, 2)
plt.imshow(f, cmap='gray')
plt.axis('off')
plt.title('Level Set edge')
curve_display(cLevelSetEdge[1, :], cLevelSetEdge[0, :], 'g.')

plt.subplot(1, 3, 3)
plt.imshow(f, cmap='gray')
plt.axis('off')
curve_display(cLevelSetRegion[1, :], cLevelSetRegion[0, :], 'g.')

plt.tight_layout()
plt.savefig('Figure1136.png')
plt.show()
