import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import scipy.io as sio

from libDIP.SnakeSegmentation import SnakeSegmentation
from libDIP.LevelSetEdgebased import LevelSetEdgebased
from libDIPUM.data_path import dip_data

# %% Parameters
# Snake
Snake = {
    'T': 0.01,
    'Sig': 11,
    'NSig': 5,
    'Mu': 0.2,
    'NIterForce': 160,
    'NIterConvergence': 35,
    'Alpha': 0.05,
    'Beta': 0.5,
    'Gamma': 2.5,
}

# Level set (edge based)
LSEdgeBased = {
    'HSize': 21,
    'Sigma': 5,
    'p': 1,
    'lambda': 50,
    'niter': 500,
}

# %% Data
img_path = dip_data('cygnusloop.tif')
f = img_as_float(imread(img_path))
if f.ndim == 3:
    f = f[..., 0]
M, N = f.shape
_ = (M, N)

# %% Initial contour
mat_snake = sio.loadmat('WorkspaceFig1138(a).mat')
xi = mat_snake['xi'].squeeze()
yi = mat_snake['yi'].squeeze()

mat_mask = sio.loadmat('WorkspaceForFig1138(b).mat')
mask = mat_mask['mask']

# %% 1) Snake
x, y, emap = SnakeSegmentation(
    f,
    xi,
    yi,
    Snake['T'],
    Snake['Sig'],
    Snake['NSig'],
    Snake['Mu'],
    Snake['NIterForce'],
    Snake['NIterConvergence'],
    Snake['Alpha'],
    Snake['Beta'],
    Snake['Gamma'],
)
_ = (x, y)

# %% 2) Level set edge based
c0, fsmooth, WBin = LevelSetEdgebased(
    f,
    mask,
    LSEdgeBased['HSize'],
    LSEdgeBased['Sigma'],
    LSEdgeBased['p'],
    LSEdgeBased['lambda'],
    LSEdgeBased['niter'],
)
_ = (c0, fsmooth)

# %% Display
plt.figure(figsize=(9, 4))

plt.subplot(1, 2, 1)
plt.imshow(emap, cmap='gray')
plt.axis('off')
plt.title('Snake edge map')

plt.subplot(1, 2, 2)
plt.imshow(WBin, cmap='gray')
plt.axis('off')
plt.title('Edge based level set')

plt.tight_layout()
plt.savefig('Figure1139.png')
plt.show()
