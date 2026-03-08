import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.imstack2vectors import imstack2vectors
from libDIPUM.principalComponents import principalComponents

print('Running Figure1238to42 (PCA on WashingtonDC multispectral stack)...')

# Parameters
PCAKeep = 2

# Data
base_dir = '/Users/michelkocher/michel/Data/DIP-DIPUM/DIP'
files = [
    'WashingtonDC-Band1-Blue-512.tif',
    'WashingtonDC-Band2-Green-512.tif',
    'WashingtonDC-Band3-Red-512.tif',
    'WashingtonDC-Band4-NearInfrared-512.tif',
    'washingtonDC-Band5-MiddleInfrared-512.tif',
    'washingtonDC-Band6-ThermalInfrared-512.tif',
]

f = []
for fn in files:
    img = imread(os.path.join(base_dir, fn))
    if img.ndim == 3:
        img = img[:, :, 0]
    f.append(img.astype(float))

S = np.stack(f, axis=2)  # (NR, NC, 6)
Size = S[:, :, 0].shape

# Stack -> vectors
X, R = imstack2vectors(S)  # (NR*NC, 6)

# Principal components (complete 6/6)
P = principalComponents(X, 6)
d = np.diag(P['Cy'])

g = np.zeros((Size[0], Size[1], 6), dtype=float)
for cpt in range(6):
    g[:, :, cpt] = np.reshape(P['Y'][:, cpt], Size, order='F')

# Principal components (partial 2/6)
P1 = principalComponents(X, PCAKeep)
d1 = np.diag(P1['Cy'])

h = np.zeros((Size[0], Size[1], 6), dtype=float)
for cpt in range(6):
    h[:, :, cpt] = np.reshape(P1['X'][:, cpt], Size, order='F')

e = h - S
MSE = np.zeros(6, dtype=float)
for cpt in range(6):
    temp = e[:, :, cpt]
    MSE[cpt] = np.sum(temp ** 2) / np.prod(Size)
TotalMSE = P1['ems']

# Console output (MATLAB style)
print('MSE =', MSE)
print('TotalMSE =', TotalMSE)

# Display 1: originals
fig1, ax1 = plt.subplots(2, 3, figsize=(12, 8))
ax1 = ax1.ravel()
info = ['visible blue', 'visible green', 'visible red', 'near infra red', 'middle infra red', 'thermal infra red']
for cpt in range(6):
    ax1[cpt].imshow(S[:, :, cpt], cmap='gray', vmin=0, vmax=255)
    ax1[cpt].set_title(f'Original, {info[cpt]}')
    ax1[cpt].axis('off')
fig1.tight_layout()
fig1.savefig('Figure1238.png')

# Display 2: covariance/eigen info
fig2, ax2 = plt.subplots(2, 3, figsize=(12, 8))
ax2 = ax2.ravel()

ax2[0].imshow(P['Cx'], cmap='gray')
ax2[0].set_title('P.Cx, complete')
ax2[0].axis('image')

ax2[1].imshow(P['Cy'], cmap='gray')
ax2[1].set_title('P.Cy, complete')
ax2[1].axis('image')

ax2[2].bar(np.arange(1, len(d) + 1), d)
ax2[2].set_title('The 6 eigenvalues')
ax2[2].axis('tight')

ax2[3].plot(P['A'].T)
ax2[3].set_title('The 6 eigenvectors')
ax2[3].axis('tight')

ax2[4].imshow(P1['Cy'], cmap='gray')
ax2[4].set_title(f'P.Cy, PCAKeep = {PCAKeep}')
ax2[4].axis('image')
# Keep a simple tick-label mimic of MATLAB code.
if P1['Cy'].shape[0] == 2:
    ax2[4].set_xticks([0, 1])
    ax2[4].set_xticklabels(['1', '100'])

ax2[5].plot(P1['A'].T)
ax2[5].set_title(f'The {PCAKeep} most significant eigenvectors')
ax2[5].axis('tight')

fig2.tight_layout()
fig2.savefig('Figure1238Bis.png')

# Display 3: PCA6 components
fig3, ax3 = plt.subplots(2, 3, figsize=(12, 8))
ax3 = ax3.ravel()
for cpt in range(6):
    ax3[cpt].imshow(g[:, :, cpt], cmap='gray')
    ax3[cpt].set_title(f'PCA6, $\\lambda_{{{cpt+1}}}$ = {d[cpt]:.3g}')
    ax3[cpt].axis('off')
fig3.tight_layout()
fig3.savefig('Figure1240.png')

# Display 4: reconstructed with q=2
fig4, ax4 = plt.subplots(2, 3, figsize=(12, 8))
ax4 = ax4.ravel()
for cpt in range(6):
    ax4[cpt].imshow(h[:, :, cpt], cmap='gray')
    ax4[cpt].set_title(f'Rec_PCA, q = {PCAKeep}')
    ax4[cpt].axis('off')
fig4.tight_layout()
fig4.savefig('Figure1241.png')

# Display 5: reconstruction errors
fig5, ax5 = plt.subplots(2, 3, figsize=(12, 8))
ax5 = ax5.ravel()
for cpt in range(6):
    ax5[cpt].imshow(e[:, :, cpt], cmap='gray')
    ax5[cpt].set_title(f'e_PCA, q = {PCAKeep}, MSE = {MSE[cpt]:.1g}')
    ax5[cpt].axis('off')
fig5.tight_layout()
fig5.savefig('Figure1242.png')

print('Saved Figure1238.png, Figure1238Bis.png, Figure1240.png, Figure1241.png, Figure1242.png')
plt.show()
