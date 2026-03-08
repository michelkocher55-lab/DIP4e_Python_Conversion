import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import ia870 as ia
from libDIPUM.skeleton import skeleton
from libDIPUM.data_path import dip_data

print('Running Figure1214 (Skeleton comparisons)...')

def _imshow_ready(x):
    """Convert ia870 color output from (C,H,W) to (H,W,C) for matplotlib."""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
        x = np.moveaxis(x, 0, -1)
    return x


# Parameters / structuring elements
Iab = ia.iahomothin()
Iab1 = ia.iaendpoints()
Bc4 = ia.iasecross()

# Data
X = imread(dip_data('blood-vessels.tif'))
if X.ndim == 3:
    X = X[:, :, 0]
X = X > 0
X = ia.ianeg(X)
# Skeleton by thinning
XThin1 = ia.iathin(X, Iab)

# Pruning
XThin2 = ia.iathin(XThin1, Iab1, 30)

# Distance transform and regional maxima
DT = ia.iadist(X, Bc4, 'EUCLIDEAN')
XThin3 = ia.iaregmax(DT, Bc4)

# Fast marching skeleton
S = skeleton(X, verbose=True)

# Display figure 1
fig1, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0, 0].imshow(X, cmap='gray')
ax[0, 0].set_title('X')
ax[0, 0].axis('off')

ax[0, 1].imshow(_imshow_ready(ia.iagshow(X, ia.iadil(XThin1))), interpolation='nearest')
ax[0, 1].set_title('Skeleton by thinning')
ax[0, 1].axis('off')

ax[1, 0].imshow(_imshow_ready(ia.iagshow(X, ia.iadil(XThin2))), interpolation='nearest')
ax[1, 0].set_title('Pruning')
ax[1, 0].axis('off')

ax[1, 1].imshow(_imshow_ready(ia.iagshow(DT, ia.iadil(XThin3, ia.iasecross(2)))), interpolation='nearest')
ax[1, 1].set_title('RMAX(DT(X))')
ax[1, 1].axis('off')

fig1.tight_layout()
fig1.savefig('Figure1214.png')

# Display figure 2 (fast marching)
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
ax2.imshow(X, cmap='gray')
ax2.set_title('Fast Marching')
ax2.set_axis_off()
ax2.invert_yaxis()  # MATLAB axis ij

rng = np.random.default_rng(0)
for L in S:
    L = np.asarray(L)
    if L.size == 0:
        continue
    color = rng.random(3)
    ax2.plot(L[:, 1], L[:, 0], '-', color=color, linewidth=1.0)

fig2.tight_layout()
fig2.savefig('Figure1214Bis.png')

print('Saved Figure1214.png and Figure1214Bis.png')
plt.show()
