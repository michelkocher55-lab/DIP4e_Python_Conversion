import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from libDIPUM.twomodegauss import twomodegauss
from libDIPUM.data_path import dip_data

# %% Fig 3.26
# Histogram specification of hidden horse image

# %% Data
f = np.array(Image.open(dip_data('hidden-horse.tif')))
if f.ndim == 3:
    f = f[..., 0]
f = f.astype(np.uint8)

# %% Specified histogram
p = twomodegauss(0.125, 0.05, 0.9, 0.03, 1, 0.05, 0.0018)
tsh = np.cumsum(p)

# %% Obtain the histogram-specified image (MATLAB histeq(f,p)-like CDF matching)
hf, _ = np.histogram(f.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
hf = hf.astype(float) / f.size
cs = np.cumsum(hf)
ct = np.cumsum(p)

mapping = np.zeros(256, dtype=np.uint8)
for r in range(256):
    # First target gray level whose CDF reaches source CDF.
    mapping[r] = np.searchsorted(ct, cs[r], side='left')

g = mapping[f]

hns, _ = np.histogram(g.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
hns = hns.astype(float) / g.size

# %% Display
fig = plt.figure(1, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(p)
plt.axis('square')
plt.axis('tight')

plt.subplot(2, 2, 2)
plt.plot(255 * tsh)
plt.axis('square')
plt.axis('tight')

plt.subplot(2, 2, 3)
plt.imshow(g, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.bar(np.arange(256), hns)
plt.axis('square')
plt.axis('tight')

plt.tight_layout()
fig.savefig('Figure326.png', dpi=150, bbox_inches='tight')
plt.show()
