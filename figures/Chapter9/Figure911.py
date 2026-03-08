import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path

# Add project root so local ia870 package can be imported when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure911

# %% Data
f = np.array(Image.open(dip_data('fingerprint-noisy.tif')))
if f.ndim == 3:
    f = f[..., 0]

# %% SE
B1 = ia.iasebox(1)

# %% Erosion
f1 = ia.iaero(f, B1)

# %% Opening
f2 = ia.iaopen(f, B1)

# %% Dilation of the opening
f3 = ia.iadil(f2, B1)

# %% Closing of the opening
f4 = ia.iaclose(f2, B1)

# %% Display
fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1)
try:
    fig.canvas.manager.set_window_title('Figure 9.11')
except Exception:
    pass

ax = ax.ravel()

ax[0].imshow(f, cmap='gray')
ax[0].set_title('f')
ax[0].axis('off')

ax[1].imshow(f1, cmap='gray')
ax[1].set_title(r'f1 = $\epsilon_{B1}(f)$')
ax[1].axis('off')

ax[2].imshow(f2, cmap='gray')
ax[2].set_title(r'f2 = $\gamma_{B1}(f)$')
ax[2].axis('off')

ax[3].imshow(f3, cmap='gray')
ax[3].set_title(r'f3 = $\delta_{B1}(f2)$')
ax[3].axis('off')

ax[4].imshow(f4, cmap='gray')
ax[4].set_title(r'f4 = $\phi_{B1}(f2)$')
ax[4].axis('off')

ax[5].axis('off')

plt.tight_layout()
fig.savefig('Figure911.png', dpi=150, bbox_inches='tight')
plt.show()
