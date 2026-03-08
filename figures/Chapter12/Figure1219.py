import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import binary_dilation

from General.mmshow import mmshow
from libDIPUM.bwboundaries import bwboundaries
from libDIPUM.bound2im import bound2im
from libDIPUM.frdescp import frdescp
from libDIPUM.ifrdescp import ifrdescp
from libDIPUM.data_path import dip_data

print('Running Figure1219 (Fourier descriptors of boundary)...')

# Data
image_path = dip_data('Fig1116(a)(chromo_binary).tif')
f = imread(image_path)
if f.ndim == 3:
    f = f[:, :, 0]
f = f > 0
NbRows, NbCols = f.shape

# Boundaries
B_list = bwboundaries(f, 8)
if not B_list:
    raise RuntimeError('No boundary found in image.')
B = B_list[0]
BIm = bound2im(B, NbRows, NbCols)
np_ = B.shape[0]

# Fourier descriptor
Z = frdescp(B)

# Inverse fourier descriptor
NbFourierDescriptor = [1434, 286, 144, 72, 36, 18, 8]
BAppIm = []
for nd in NbFourierDescriptor:
    nd_use = min(nd, len(Z))
    if nd_use % 2 == 1:
        nd_use -= 1
    BApp = ifrdescp(Z, nd_use)
    BAppIm.append(bound2im(BApp, NbRows, NbCols))

# Display 1
IxCenter = int(np.round(len(B) / 2.0) + 1)
omega = ((np.arange(1, len(B) + 1) - IxCenter) / IxCenter) * np.pi

fig1, ax = plt.subplots(2, 2, figsize=(11, 8))

plt.sca(ax[0, 0])
mmshow(f.astype(float), binary_dilation(BIm), binary_dilation(BIm))
ax[0, 0].set_title('f, Boundary(f)')
ax[0, 0].axis('off')

ax[0, 1].plot(np.arange(1, np_ + 1), B[:, 0], 'r', np.arange(1, np_ + 1), B[:, 1], 'g')
ax[0, 1].set_xlabel('k')
ax[0, 1].set_title('x[k], y[k]')
ax[0, 1].axis('tight')

with np.errstate(divide='ignore'):
    mag_db = 20.0 * np.log10(np.abs(Z))
ax[1, 0].plot(omega, mag_db)
ax[1, 0].set_title('|Z(ω)|, Z(ω) = fft(B2[k])')
ax[1, 0].set_xlabel('ω')
ax[1, 0].axis('tight')

ax[1, 1].plot(omega, np.unwrap(np.angle(Z)) * 180.0 / np.pi)
ax[1, 1].set_title('∠(Z(ω))')
ax[1, 1].set_xlabel('ω')
ax[1, 1].axis('tight')

fig1.tight_layout()
fig1.savefig('Figure1219.png')

# Display 2
fig2, ax2 = plt.subplots(2, 4, figsize=(14, 7))
ax2 = ax2.ravel()

ax2[0].imshow(BIm, cmap='gray', interpolation='nearest')
ax2[0].set_title('Boundary (f)')
ax2[0].axis('off')

for cpt, nd in enumerate(NbFourierDescriptor, start=1):
    plt.sca(ax2[cpt])
    mmshow(BIm.astype(float), BAppIm[cpt - 1], BAppIm[cpt - 1])
    ax2[cpt].set_title(f'IFD {min(nd, len(Z))}/{len(Z)}')
    ax2[cpt].axis('off')

# Approximate MATLAB linkaxes behavior.
xlim = ax2[0].get_xlim()
ylim = ax2[0].get_ylim()
for a in ax2[1:]:
    a.set_xlim(xlim)
    a.set_ylim(ylim)

fig2.tight_layout()
fig2.savefig('Figure1219Bis.png')

print('Saved Figure1219.png and Figure1219Bis.png')
plt.show()
