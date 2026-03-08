import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import sys
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure931

# %% Parameters
Bc4 = ia.iasecross()

# %% Data (mask)
img_name = dip_data('text-image.tif')
g = np.array(Image.open(img_name))

if g.ndim == 3:
    g = g[..., 0]
g = g > 0

# %% Erosion
B = ia.iaseline(41, 90)
X1 = ia.iaero(g, B)

# %% Opening (marker)
f0 = ia.iaopen(g, B)

# %% Reconstruction by dilation
try:
    _raw = input('Fast (1) or iterative (2) : ').strip()
    Choix = int(_raw) if _raw else 1
except Exception:
    Choix = 1

f_iters = [f0]

if Choix == 1:
    X3 = ia.iainfrec(f0, g, Bc4)
elif Choix == 2:
    k = 0
    while True:
        nxt = ia.iaintersec(ia.iadil(f_iters[k], Bc4), g)
        f_iters.append(nxt)
        if np.array_equal(f_iters[k], f_iters[k + 1]):
            break
        else:
            k += 1
    X3 = f_iters[k]
    print(k + 1)
else:
    raise ValueError('Plouc')

# %% Display
fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, num=1)
try:
    fig.canvas.manager.set_window_title('Figure9.31')
except Exception:
    pass

ax = ax.ravel()

ax[0].imshow(g, cmap='gray')
ax[0].set_title('g')
ax[0].axis('off')

ax[1].imshow(X1, cmap='gray')
ax[1].set_title(r'X1 = $\epsilon_B(g)$')
ax[1].axis('off')

ax[2].imshow(f0, cmap='gray')
ax[2].set_title(r'X1 = $\gamma_B(g)$')
ax[2].axis('off')

ax[3].imshow(X3, cmap='gray')
ax[3].set_title(r'X3 = $R^{\delta}(X1, g)$')
ax[3].axis('off')

plt.tight_layout()
fig.savefig('Figure931.png', dpi=150, bbox_inches='tight')
plt.show()
