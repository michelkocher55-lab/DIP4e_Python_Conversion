import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import scipy.io as sio

from libDIP.intScaling4e import intScaling4e
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e

from libDIP.levelSetForce4e import levelSetForce4e
from libDIP.levelSetIterate4e import levelSetIterate4e
from libDIP.levelSetReInit4e import levelSetReInit4e

from libDIPUM.contourc import contourc
from libDIPUM.curve_display import curve_display
from libDIPUM.snake_display import snake_display
from libDIPUM.data_path import dip_data

# (1) Snake part (gvf)
mat = sio.loadmat('Figure118.mat')
f = mat['f']
xi = mat['yi'].squeeze()
yi = mat['xi'].squeeze()
M, N = f.shape

T = 0.005
Sig = 11
NSig = 5
Alpha = 10 * 0.05
Beta = 0.5
Gamma = 5

emap = snakeMap4e(f, T, Sig, NSig, 'both')
emap = img_as_float(intScaling4e(emap))

FTxb, FTyb = snakeForce4e(emap, 'gvf', 0.2, 160)
magb = np.sqrt(FTxb**2 + FTyb**2)
FTxb = FTxb / (magb + 1e-10)
FTyb = FTyb / (magb + 1e-10)

x = xi.copy()
y = yi.copy()
for _ in range(400):
    x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTxb, FTyb)
    x, y = snakeReparam4e(x, y)
x, y = snakeReparam4e(x, y)

# (2) Level Set (Edge based)
f1 = img_as_float(imread(dip_data('rose479by512.tif')))
M1, N1 = f1.shape
y1, x1 = np.meshgrid(np.arange(1, N1 + 1), np.arange(1, M1 + 1))
center = (int(round(M1 / 2)), int(round(N1 / 2)))
r = max(center) - max(center) / 3.0
phi0 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2) - r

W = levelSetForce4e('gradient', [f1, 1, 50])
T = (np.max(W) + np.min(W)) / 2.0
W = W > T

phi = phi0.copy()
C = 0.5
for i in range(1, 800 + 1):
    F = levelSetForce4e('geodesic', [phi, C, W])
    phi = levelSetIterate4e(phi, F)
    if i % 5 == 0:
        phi = levelSetReInit4e(phi, 5, 0.5)
c = contourc(phi, [0, 0])

# (3) Level Set (Region based) Chan-Vese
mu = 0.5
nu = 0
lambda1 = 1
lambda2 = 1
phi = phi0.copy()
for i in range(1, 800 + 1):
    F = levelSetForce4e('regioncurve', [f1, phi, mu, nu, lambda1, lambda2], ['Fn', 'Cn'])
    phi = levelSetIterate4e(phi, F)
    if i % 5 == 0:
        phi = levelSetReInit4e(phi, 5, 0.5)
c1 = contourc(phi, [0, 0])

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(f, cmap='gray')
axes[0].axis('off')
plt.sca(axes[0])
snake_display(x, y, 'g.')
axes[0].set_title('snake')

axes[1].imshow(f1, cmap='gray')
axes[1].axis('off')
curve_display(c[1, :], c[0, :], 'g.', ax=axes[1])
axes[1].set_title('levelset geodesic')

axes[2].imshow(f1, cmap='gray')
axes[2].axis('off')
curve_display(c1[1, :], c1[0, :], 'g.', ax=axes[2])
axes[2].set_title('levelset region based')

plt.tight_layout()
plt.savefig('Figure1135.png')
plt.show()
