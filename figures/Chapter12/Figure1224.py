import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

print('Running Figure1224 (Americas at night ratios)...')

# Data
base_dir = '/Users/michelkocher/michel/Data/DIP-DIPUM/DIP'
Names = ['americas-at-night1', 'americas-at-night2', 'americas-at-night3', 'americas-at-night4']
OtherNames = ['Canada', 'USA', 'Central America', 'South America']

f = []
X1 = []
X2 = []
Total = 0

for name in Names:
    path = os.path.join(base_dir, f'{name}.tif')
    img = imread(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    f.append(img)

    x1, x2 = np.where(img != 0)  # Light is there
    X1.append(x1)
    X2.append(x2)
    Total += x1.size

# Ratio
Ratio = np.zeros(len(Names), dtype=float)
for i in range(len(Names)):
    Ratio[i] = X1[i].size / Total if Total > 0 else 0.0

# Display
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.ravel()

for i in range(len(Names)):
    ax[i].imshow(f[i], cmap='gray')
    ax[i].set_title(f"{OtherNames[i]} {100*Ratio[i]:.6g} % of the energy is spent here")
    ax[i].axis('off')

fig.tight_layout()
fig.savefig('Figure1224.png')

print('Saved Figure1224.png')
plt.show()
