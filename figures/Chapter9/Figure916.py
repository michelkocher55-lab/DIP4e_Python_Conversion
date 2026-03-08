import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import sys
from pathlib import Path
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure916

# %% Data
f = np.array(Image.open(dip_data('lincoln.tif')))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
B1 = ia.iasebox(1)
B0 = ia.iasebox(0)

# %% Erosion
f1 = ia.iaero(f, B1)

# %% Inner gradient
f2 = ia.iagradm(f, B0, B1)

# %% Outer gradient
f3 = ia.iagradm(f, B1, B0)

# %% Beucher gradient
f4 = ia.iagradm(f, B1, B1)

# %% Display
fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, num=1)
try:
    fig.canvas.manager.set_window_title('Figure 9.16')
except Exception:
    pass

ax = ax.ravel()

ax[0].imshow(f, cmap='gray')
ax[0].set_title('f')
ax[0].axis('off')

ax[1].imshow(f2, cmap='gray')
ax[1].set_title(r'f2 = $f - \epsilon_{B1}(f)$')
ax[1].axis('off')

ax[2].imshow(f3, cmap='gray')
ax[2].set_title(r'f3 = $\delta_{B1}(f) - f$')
ax[2].axis('off')

ax[3].imshow(f4, cmap='gray')
ax[3].set_title(r'f4 = $\delta_{B1}(f) - \epsilon_{B1}(f)$')
ax[3].axis('off')

plt.tight_layout()
fig.savefig('Figure916.png', dpi=150, bbox_inches='tight')
plt.show()
