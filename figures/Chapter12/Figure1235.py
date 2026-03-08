import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.specxture import specxture

print('Running Figure1235 (Matches and spectra)...')

# Data
base_dir = '/Users/michelkocher/michel/Data/DIP-DIPUM/DIP'
f1 = imread(os.path.join(base_dir, 'matches-random.tif'))
f2 = imread(os.path.join(base_dir, 'matches-aligned.tif'))

if f1.ndim == 3:
    f1 = f1[:, :, 0]
if f2.ndim == 3:
    f2 = f2[:, :, 0]

# Spectrum (cartesian mode)
F1 = np.fft.fft2(f1)
F2 = np.fft.fft2(f2)

# Spectrum (polar mode)
F1Rho, F1Theta, _ = specxture(f1)
F2Rho, F2Theta, _ = specxture(f2)

# Display
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0, 0].imshow(f1, cmap='gray')
ax[0, 0].set_title('f_1[k, l]')
ax[0, 0].axis('off')

ax[0, 1].imshow(f2, cmap='gray')
ax[0, 1].set_title('f_2[k, l]')
ax[0, 1].axis('off')

ax[1, 0].imshow(np.log10(np.abs(np.fft.fftshift(F1))), cmap='gray')
ax[1, 0].set_title('|F_1[m, n]|')
ax[1, 0].axis('off')

ax[1, 1].imshow(np.log10(np.abs(np.fft.fftshift(F2))), cmap='gray')
ax[1, 1].set_title('|F_2[m, n]|')
ax[1, 1].axis('off')

fig.tight_layout()
fig.savefig('Figure1235.png')

print('Saved Figure1235.png')
plt.show()
