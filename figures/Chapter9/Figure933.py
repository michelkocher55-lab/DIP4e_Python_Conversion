import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import sys
from pathlib import Path
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure933
# Hole closing without marker 9.5-28, 9.5-29

# %% Init
Fig = 1

# %% Data
img_name = dip_data('text-image.tif')
Mask = imread(img_name)

if Mask.ndim == 3:
    Mask = Mask[..., 0]
Mask = Mask > 0

MaskC = ia.ianeg(Mask)

# %% Marker
Frame = np.zeros_like(Mask, dtype=bool)
Frame[0, :] = True
Frame[-1, :] = True
Frame[:, 0] = True
Frame[:, -1] = True

Marker1 = ia.iaintersec(Frame, Mask)
Marker2 = ia.iaintersec(np.logical_not(Marker1), Frame)

# %% Reconstruction
Temp = ia.iainfrec(Marker2, MaskC, ia.iasecross(1))
FillHole = ia.ianeg(Temp)

# %% Display (Figure 1)
fig1, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, num=Fig)
Fig += 1
ax = ax.ravel()

ax[0].imshow(Mask, cmap='gray')
ax[0].set_title('Mask')
ax[0].axis('off')

ax[1].imshow(Frame, cmap='gray')
ax[1].set_title('Frame')
ax[1].axis('off')

ax[2].imshow(Marker1, cmap='gray')
ax[2].set_title('Marker1 = Frame ∩ Mask')
ax[2].axis('off')

ax[3].imshow(ia.ianeg(Marker1), cmap='gray')
ax[3].set_title('not Marker1')
ax[3].axis('off')

fig1.tight_layout()

# %% Display (Figure 2)
fig2, bx = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, num=Fig)
bx = bx.ravel()

bx[0].imshow(ia.iadil(Marker2), cmap='gray')
bx[0].set_title('Marker2 = not(Marker1) and Frame')
bx[0].axis('off')

bx[1].imshow(MaskC, cmap='gray')
bx[1].set_title('Mask complement')
bx[1].axis('off')

bx[2].imshow(Temp, cmap='gray')
bx[2].set_title('Reconstruction by dilation')
bx[2].axis('off')

bx[3].imshow(FillHole, cmap='gray')
bx[3].set_title('Complement of reconstruction')
bx[3].axis('off')

fig2.tight_layout()

# %% Print
fig1.savefig('Figure933.png', dpi=150, bbox_inches='tight')
fig2.savefig('Figure933Bis.png', dpi=150, bbox_inches='tight')

plt.show()
