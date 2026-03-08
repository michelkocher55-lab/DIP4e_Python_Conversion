import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path

# Add project root so local packages can be imported when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ia870 as ia
from General.AddSE2Image import AddSE2Image
from libDIPUM.data_path import dip_data

# %% Figure941
# Morphological gradient

# %% Data
f = np.array(Image.open(dip_data('headCT.tif')))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
B = ia.iasedisk(2)

# %% Morphological gradients
g1 = ia.iadil(f, B)
g2 = ia.iaero(f, B)
g3 = ia.iasubm(g1, g2)  # Beucher gradient
g4 = ia.iasubm(f, g2)   # Internal gradient
g5 = ia.iasubm(g1, f)   # External gradient

# %% Display
fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1)
try:
    fig.canvas.manager.set_window_title('Figure 9.39')
except Exception:
    pass

ax = ax.ravel()

ax[0].imshow(AddSE2Image(f, ia.iasedisk(2), int(np.max(f))), cmap='gray')
ax[0].set_title('f, B')
ax[0].axis('off')

ax[1].imshow(g1, cmap='gray')
ax[1].set_title('g1 = dilation of f')
ax[1].axis('off')

ax[2].imshow(g2, cmap='gray')
ax[2].set_title('g2 = erosion of f')
ax[2].axis('off')

ax[3].imshow(g3, cmap='gray')
ax[3].set_title('Beucher gradient = g1 - g2')
ax[3].axis('off')

ax[4].imshow(g4, cmap='gray')
ax[4].set_title('Internal gradient = f - g2')
ax[4].axis('off')

ax[5].imshow(g5, cmap='gray')
ax[5].set_title('External gradient = g1 - f')
ax[5].axis('off')

plt.tight_layout()
fig.savefig('Figure941.png', dpi=150, bbox_inches='tight')
plt.show()
