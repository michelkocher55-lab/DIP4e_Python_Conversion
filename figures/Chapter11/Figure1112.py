import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.io import loadmat, savemat
from libDIPUM.snake_manual_input import snake_manual_input
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIPUM.snake_display import snake_display
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Parameters
T = 0.005
Sig = 11
NSig = 5
r = 10
NIter = 100

# Data
img_path = dip_data('rose957by1024.tif')
f = imread(img_path)

if os.path.exists('Figure118.mat'):
    mat = loadmat('Figure118.mat')
    xi = mat['xi'].squeeze()
    yi = mat['yi'].squeeze()
else:
    xi, yi = snake_manual_input(f, 150, 'ro')
    savemat('Figure118.mat', {'xi': xi, 'yi': yi})

# Edge map
emap = snakeMap4e(f, T, Sig, NSig, 'both')

# Scale to [0,1]
emap = intScaling4e(emap)

# Snake force (gradient)
FTxa, FTya = snakeForce4e(emap, 'gradient')

# Normalize
maga = np.sqrt(FTxa ** 2 + FTya ** 2)
FTxa = FTxa / (maga + 1e-10)
FTya = FTya / (maga + 1e-10)

# Snake force (GVF)
FTxb, FTyb = snakeForce4e(emap, 'gvf', 0.2, 160)

# Normalize
magb = np.sqrt(FTxb ** 2 + FTyb ** 2)
FTxb = FTxb / (magb + 1e-10)
FTyb = FTyb / (magb + 1e-10)

xa = xi.copy()
ya = yi.copy()
xb = xi.copy()
yb = yi.copy()

# Iterate
for _ in range(150):
    xa, ya = snakeIterate4e(1, 0.5, 5, xa, ya, 1, FTxa, FTya)
    xa, ya = snakeReparam4e(xa, ya)

    xb, yb = snakeIterate4e(1, 0.5, 5, xb, yb, 1, FTxb, FTyb)
    xb, yb = snakeReparam4e(xb, yb)

xa, ya = snakeReparam4e(xa, ya)
xb, yb = snakeReparam4e(xb, yb)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].quiver(
    np.flipud(FTya[::r, ::r]),
    np.flipud(-FTxa[::r, ::r]),
    angles='xy',
    scale_units='xy',
    scale=1
)
axes[0, 0].axis('off')

axes[1, 0].imshow(f, cmap='gray')
axes[1, 0].axis('off')
plt.sca(axes[1, 0])
snake_display(xa, ya, 'g.')

axes[0, 1].quiver(
    np.flipud(FTyb[::r, ::r]),
    np.flipud(-FTxb[::r, ::r]),
    angles='xy',
    scale_units='xy',
    scale=1
)
axes[0, 1].axis('off')

axes[1, 1].imshow(f, cmap='gray')
axes[1, 1].axis('off')
plt.sca(axes[1, 1])
snake_display(xb, yb, 'g.')

plt.tight_layout()
plt.savefig('Figure1112.png')
plt.show()
