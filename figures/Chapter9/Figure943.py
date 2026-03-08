import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
import sys
from pathlib import Path
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure943
# Granulometry

# %% Data
f = np.array(Image.open(dip_data('wood-dowels.tif')))
if f.ndim == 3:
    f = f[..., 0]

# %% Smoothing (fspecial gaussian + imfilter equivalent)
HSize = 5
Sigma = 1
ax = np.arange(-(HSize // 2), HSize // 2 + 1)
xx, yy = np.meshgrid(ax, ax)
h = np.exp(-(xx**2 + yy**2) / (2.0 * Sigma**2))
h = h / np.sum(h)

g = convolve(f.astype(float), h, mode='reflect')
g = np.clip(np.round(g), 0, 255).astype(f.dtype)

# %% Opening
g1 = np.zeros((g.shape[0], g.shape[1], 35), dtype=g.dtype)
LesArea = np.zeros(35, dtype=float)
for cpt in range(1, 36):
    opened = ia.iaopen(g, ia.iasedisk(cpt))
    g1[:, :, cpt - 1] = opened
    LesArea[cpt - 1] = float(np.sum(opened))

# %% Display figure 1
LesRadii = [10, 20, 25, 30]
fig1, axarr = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1)
try:
    fig1.canvas.manager.set_window_title('Figure 9.41')
except Exception:
    pass

ax = axarr.ravel()

ax[0].imshow(f, cmap='gray')
ax[0].set_title('f')
ax[0].axis('off')

ax[1].imshow(g, cmap='gray')
ax[1].set_title(f'g = gaussian smooth, sigma={Sigma}, size={HSize}')
ax[1].axis('off')

for i, r in enumerate(LesRadii):
    ax[2 + i].imshow(g1[:, :, r - 1], cmap='gray')
    ax[2 + i].set_title(f'g{r} = opening with disk radius {r}')
    ax[2 + i].axis('off')

fig1.tight_layout()
fig1.savefig('Figure943.png', dpi=150, bbox_inches='tight')

# %% Display figure 2
fig2, bx = plt.subplots(1, 2, figsize=(12, 4), num=2)
try:
    fig2.canvas.manager.set_window_title('Figure 9.41 bis')
except Exception:
    pass

bx[0].bar(np.arange(1, 36), LesArea)
bx[0].set_xlabel('Radius')
bx[0].set_title('Volume versus radius')
bx[0].axis('tight')
bx[0].set_box_aspect(1)

dvol = -np.diff(LesArea)
bx[1].bar(np.arange(1, len(dvol) + 1), dvol)
bx[1].set_xlabel('Radius')
bx[1].set_title('Negative derivative of volume')
bx[1].axis('tight')
bx[1].set_box_aspect(1)

fig2.tight_layout()
fig2.savefig('Figure943Bis.png', dpi=150, bbox_inches='tight')

plt.show()
