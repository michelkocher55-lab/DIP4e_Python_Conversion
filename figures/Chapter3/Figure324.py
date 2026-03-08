import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from libDIPUM.data_path import dip_data

# %% Figure 3.24
# Image of hidden horse and its histogram

# %% Data
f = np.array(Image.open(dip_data('hidden-horse.tif')))
if f.ndim == 3:
    f = f[..., 0]

# %% Obtain normalized histogram of the image.
counts, _ = np.histogram(f.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
pf = counts.astype(float) / f.size

# %% Display
fig = plt.figure(1, figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(f, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(np.arange(256), pf)
#plt.axis('square')

plt.tight_layout()
fig.savefig('Figure324.png', dpi=150, bbox_inches='tight')
plt.show()
