import matplotlib.pyplot as plt
from skimage.io import imread
from libDIP.intScaling4e import intScaling4e
from libDIPUM.unsharp import unsharp
from libDIPUM.data_path import dip_data

# Parameters
k = [1, 2, 3]
N = 31
Sigma = 5.0

# Data
f = imread(dip_data('girl-blurred.tif'))
if f.ndim == 3:
    f = f[:, :, 0]

# Unsharp masking (k = 1)
g, gb, gmask_raw = unsharp(f, k[0], N, Sigma)
gmask = intScaling4e(gmask_raw)

# Highboost filtering
ghb2, _, _ = unsharp(f, k[1], N, Sigma)
ghb3, _, _ = unsharp(f, k[2], N, Sigma)

# Display
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(f, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gb, cmap='gray', vmin=0, vmax=1)
plt.title('blurred')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(gmask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(g, cmap='gray', vmin=0, vmax=1)
plt.title(f'unsharp, k = {k[0]}')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(ghb2, cmap='gray', vmin=0, vmax=1)
plt.title(f'unsharp, k = {k[1]}')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(ghb3, cmap='gray', vmin=0, vmax=1)
plt.title(f'unsharp, k = {k[2]}')
plt.axis('off')

plt.tight_layout()
plt.savefig('Figure355.png')
print('Saved Figure355.png')
plt.show()
