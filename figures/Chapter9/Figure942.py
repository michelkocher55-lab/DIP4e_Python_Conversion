import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu
import sys
from pathlib import Path
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure942
# Rice grains statistics

# %% Data
f = np.array(Image.open(dip_data('rice-shaded.tif')))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
B = ia.iasedisk(40, '2D', 'OCTAGON')

# %% Top Hat
g1 = ia.iaopen(f, B)
g2 = ia.iasubm(f, g1)

# %% Thresholding
level = threshold_otsu(f)
X = f > level

level1 = threshold_otsu(g2)
Y = g2 > level1

# %% Noise cleaning
Y1 = ia.iaclose(ia.iaopen(Y, ia.iasecross()), ia.iasecross())

# %% Edge off
Y2 = ia.iaedgeoff(Y, ia.iasecross())

# %% Labeling
Yl = ia.ialabel(Y2, ia.iasebox())

# %% Statistics
Area = ia.iablob(Yl, 'area', 'data')

# %% Display figure 1
fig1, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1)
ax = ax.ravel()

ax[0].imshow(f, cmap='gray')
ax[0].set_title('f')
ax[0].axis('off')

ax[1].imshow(X, cmap='gray')
ax[1].set_title(f'X threshold on f, Otsu = {int(round(level))}')
ax[1].axis('off')

ax[2].axis('off')

ax[3].imshow(g1, cmap='gray')
ax[3].set_title('g1 = opening with disk radius 40')
ax[3].axis('off')

ax[4].imshow(g2, cmap='gray')
ax[4].set_title('g2 = f - g1')
ax[4].axis('off')

ax[5].imshow(Y, cmap='gray')
ax[5].set_title(f'Y threshold on g2, Otsu = {int(round(level1))}')
ax[5].axis('off')

fig1.tight_layout()
fig1.savefig('Figure942.png', dpi=150, bbox_inches='tight')

# %% Display figure 2
fig2, bx = plt.subplots(2, 2, figsize=(10, 8), num=2)
bx = bx.ravel()

bx[0].imshow(Y1, cmap='gray')
bx[0].set_title('Cleaning by ASF')
bx[0].axis('off')

bx[1].imshow(Y2, cmap='gray')
bx[1].set_title('Edge off')
bx[1].axis('off')

lbl_img = np.asarray(ia.iaglblshow(Yl))
if lbl_img.ndim == 3 and lbl_img.shape[0] in (3, 4):
    # ia870 may return channel-first (C, H, W); convert to (H, W, C).
    lbl_img = np.transpose(lbl_img, (1, 2, 0))
bx[2].imshow(lbl_img)
bx[2].set_title(f'Labeling, number of objects = {int(np.max(Yl))}')
bx[2].axis('off')

area_vals = np.asarray(Area).ravel().astype(float)
bx[3].hist(area_vals, bins=20)
bx[3].set_title('Area distribution')

fig2.tight_layout()
fig2.savefig('Figure942Bis.png', dpi=150, bbox_inches='tight')

plt.show()
