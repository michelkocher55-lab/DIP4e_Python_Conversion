import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import ia870 as ia
from libDIPUM.data_path import dip_data

print('Running Figure1228 (Morphology + labeling + skeleton)...')

# Init
Bc8 = ia.iasebox()

# Data
image_path = dip_data('WashingtonDC-Band4-NearInfrared-512.tif')
f = imread(image_path)
if f.ndim == 3:
    f = f[:, :, 0]

# Erosion
fe = ia.iaero(f, ia.iasedisk(1))

# Thresholding (as in MATLAB script)
T = 65
X = fe < T

# Labeling
XLabel = ia.ialabel(X, Bc8)
NConnComp = int(np.max(XLabel))
XAreaLabel = ia.iablob(XLabel, 'AREA', 'image')

# Keeping the largest connected component
Y = XAreaLabel >= np.max(XAreaLabel)

# Skeleton
Sk = ia.iathin(Y, ia.iahomothin(), -1, 45, 'CLOCKWISE')

# Display
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.ravel()

ax[0].imshow(f, cmap='gray')
ax[0].set_title('f')
ax[0].axis('off')

ax[1].imshow(fe, cmap='gray')
ax[1].set_title('fe = ε_B(f)')
ax[1].axis('off')

ax[2].imshow(X, cmap='gray')
ax[2].set_title(f'X = fe < {T}')
ax[2].axis('off')

ax[3].imshow(np.moveaxis(ia.iaglblshow(XLabel), 0, -1))
ax[3].set_title(f'label(X, Bc_8), N_CC = {NConnComp}')
ax[3].axis('off')

ax[4].imshow(Y, cmap='gray')
ax[4].set_title(f'Largest CC, A = {int(np.sum(Y))}')
ax[4].axis('off')

ax[5].imshow(Sk, cmap='gray')
ax[5].set_title('Sk = skel(Y)')
ax[5].axis('off')

# linkaxes approximation
xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()
for a in ax[1:]:
    a.set_xlim(xlim)
    a.set_ylim(ylim)

fig.tight_layout()
fig.savefig('Figure1228.png')

print('Saved Figure1228.png')
plt.show()
