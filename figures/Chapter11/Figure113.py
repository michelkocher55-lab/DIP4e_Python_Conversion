import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.io import loadmat
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIPUM.snake_display import snake_display
from libDIPUM.data_path import dip_data

# Parameters
Alpha = 0.5
Beta = 2.0
Gamma = 5.0
NPoints = 150
T = 0.001
Sig = 15
NSig = 3
Order = 'both'
NIter = 300

# Data
img_path = dip_data('noisy-elliptical-object.tif')
g = imread(img_path)

# to work with the same initial snake
mat = loadmat('Figure112.mat')
xi = mat['xi'].squeeze()
yi = mat['yi'].squeeze()


def process(g, T, Sig, NSig, NIter, mode, Alpha, Beta, Gamma, xi, yi):
    emap = snakeMap4e(g, T, Sig, NSig, mode)

    # Snake force using plain gradient.
    FTx, FTy = snakeForce4e(emap, 'gradient')

    # Normalize the forces.
    mag = np.sqrt(FTx ** 2 + FTy ** 2)
    FTx = FTx / (mag + 1e-10)
    FTy = FTy / (mag + 1e-10)

    x = xi.copy()
    y = yi.copy()

    # Iterate NIter times
    for _ in range(NIter):
        x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
        x, y = snakeReparam4e(x, y)

    return emap, x, y


# (1) smooth, edge, smooth
emap1, x1, y1 = process(g, T, Sig, NSig, NIter, 'both', Alpha, Beta, Gamma, xi, yi)

# (2) edge, smooth
emap2, x2, y2 = process(g, T, Sig, NSig, NIter, 'after', Alpha, Beta, Gamma, xi, yi)

# (3) smooth, edge
emap3, x3, y3 = process(g, T, Sig, NSig, NIter, 'before', Alpha, Beta, Gamma, xi, yi)

# (4) edge
emap4, x4, y4 = process(g, T, Sig, NSig, NIter, 'none', Alpha, Beta, Gamma, xi, yi)

# Show results
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
axes1[0, 0].imshow(emap1, cmap='gray')
axes1[0, 0].set_title('smooth, edge, smooth')
axes1[0, 0].axis('off')

axes1[1, 0].imshow(g, cmap='gray')
axes1[1, 0].axis('off')
plt.sca(axes1[1, 0])
snake_display(x1, y1, 'g.')

axes1[0, 1].imshow(emap2, cmap='gray')
axes1[0, 1].set_title('edge, smooth')
axes1[0, 1].axis('off')

axes1[1, 1].imshow(g, cmap='gray')
axes1[1, 1].axis('off')
plt.sca(axes1[1, 1])
snake_display(x2, y2, 'g.')

plt.tight_layout()
plt.savefig('Figure113.png')

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
axes2[0, 0].imshow(emap3, cmap='gray')
axes2[0, 0].set_title('smooth, edge')
axes2[0, 0].axis('off')

axes2[1, 0].imshow(g, cmap='gray')
axes2[1, 0].axis('off')
plt.sca(axes2[1, 0])
snake_display(x3, y3, 'g.')

axes2[0, 1].imshow(emap4, cmap='gray')
axes2[0, 1].set_title('edge')
axes2[0, 1].axis('off')

axes2[1, 1].imshow(g, cmap='gray')
axes2[1, 1].axis('off')
plt.sca(axes2[1, 1])
snake_display(x4, y4, 'g.')

plt.tight_layout()
plt.savefig('Figure113Bis.png')
plt.show()
