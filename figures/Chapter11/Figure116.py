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
T = 0.001
Sig = 15
NSig = 3
NIter = 265

# Data
img_path = dip_data('noisy-elliptical-object.tif')
g = imread(img_path)

# Load initial snake
mat = loadmat('Figure112.mat')
xi = mat['xi'].squeeze()
yi = mat['yi'].squeeze()

# Edge map
emap = snakeMap4e(g, T, Sig, NSig, 'both')

# Snake force
FTx, FTy = snakeForce4e(emap, 'gradient')

# Normalize forces
mag = np.sqrt(FTx ** 2 + FTy ** 2)
FTx = FTx / (mag + 1e-10)
FTy = FTy / (mag + 1e-10)

# Without reparametrization
x1 = xi.copy()
y1 = yi.copy()
for _ in range(NIter):
    x1, y1 = snakeIterate4e(Alpha, Beta, Gamma, x1, y1, 1, FTx, FTy)

# Reparametrization after last iteration
x2 = xi.copy()
y2 = yi.copy()
for _ in range(NIter):
    x2, y2 = snakeIterate4e(Alpha, Beta, Gamma, x2, y2, 1, FTx, FTy)
x2, y2 = snakeReparam4e(x2, y2)

# Reparametrization every ten iterations
x3 = xi.copy()
y3 = yi.copy()
for i in range(1, NIter + 1):
    x3, y3 = snakeIterate4e(Alpha, Beta, Gamma, x3, y3, 1, FTx, FTy)
    if i % 10 == 0:
        x3, y3 = snakeReparam4e(x3, y3)

# Reparametrization every iteration
x4 = xi.copy()
y4 = yi.copy()
for _ in range(NIter):
    x4, y4 = snakeIterate4e(Alpha, Beta, Gamma, x4, y4, 1, FTx, FTy)
    x4, y4 = snakeReparam4e(x4, y4)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(g, cmap='gray')
axes[0, 0].axis('off')
plt.sca(axes[0, 0])
snake_display(x1, y1, 'g.')

axes[0, 1].imshow(g, cmap='gray')
axes[0, 1].axis('off')
plt.sca(axes[0, 1])
snake_display(x2, y2, 'g.')

axes[1, 0].imshow(g, cmap='gray')
axes[1, 0].axis('off')
plt.sca(axes[1, 0])
snake_display(x3, y3, 'g.')

axes[1, 1].imshow(g, cmap='gray')
axes[1, 1].axis('off')
plt.sca(axes[1, 1])
snake_display(x4, y4, 'g.')

plt.tight_layout()
plt.savefig('Figure116.png')
plt.show()
