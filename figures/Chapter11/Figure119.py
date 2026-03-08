import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIP.snakeMap4e import snakeMap4e
from libDIP.snakeForce4e import snakeForce4e
from libDIP.snakeIterate4e import snakeIterate4e
from libDIP.snakeReparam4e import snakeReparam4e
from libDIPUM.snake_display import snake_display
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data('U200.tif')
f = imread(img_path)

# Parameters
T = 0.001
Sig = 3
NSig = 1
NIter = 1000
Alpha = 0.06
Beta = 0
Gamma = 1

# Specify a circle
t = np.arange(0, 2 * np.pi + 0.1, 0.1)
xi = 100 + 80 * np.cos(t)
yi = 100 + 80 * np.sin(t)

# Close the snake
xi = np.concatenate([xi, [xi[0]]])
yi = np.concatenate([yi, [yi[0]]])

# Edge map
emap = snakeMap4e(f, T, Sig, NSig, 'after')

# Scale to range [0,1]
emap = intScaling4e(emap)

# Snake force
FTx, FTy = snakeForce4e(emap, 'gradient')

# Normalize it
mag = np.sqrt(FTx ** 2 + FTy ** 2)
FTx = FTx / (mag + np.finfo(float).eps)
FTy = FTy / (mag + np.finfo(float).eps)

# Process
x = xi.copy()
y = yi.copy()

for _ in range(NIter):
    x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
    x, y = snakeReparam4e(x, y)

# Redistribute one last time
x, y = snakeReparam4e(x, y)

# Display figure 1
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
axes2[0, 0].imshow(f, cmap='gray')
axes2[0, 0].axis('off')
axes2[0, 0].plot(np.append(yi, yi[0]), np.append(xi, xi[0]), 'k.')

axes2[0, 1].imshow(emap, cmap='gray')
axes2[0, 1].axis('off')

axes2[1, 0].quiver(np.flipud(FTy[::2, ::2]), np.flipud(-FTx[::2, ::2]))
axes2[1, 0].axis('off')

axes2[1, 1].imshow(f, cmap='gray')
axes2[1, 1].axis('off')
plt.sca(axes2[1, 1])
snake_display(x, y, 'g.')

plt.tight_layout()
plt.savefig('Figure118.png')
plt.show()
