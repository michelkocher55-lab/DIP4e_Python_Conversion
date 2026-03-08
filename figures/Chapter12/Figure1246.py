import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from skimage.util import random_noise

print('Running Figure1246 (Spread of derivatives)...')

# Data
fconstOrig = 0.75 * np.ones((600, 600), dtype=float)

fedgeOrig = 0.75 * np.ones((600, 600), dtype=float)
fedgeOrig[300:, :] = 0.0  # MATLAB: end/2:end

fcrOrig = 0.75 * np.ones((600, 600), dtype=float)
fcrOrig[300:, :300] = 0.0  # MATLAB: end/2:end,1:end/2

# Add noise (Gaussian variance = .003)
fconst = random_noise(fconstOrig, mode='gaussian', mean=0.0, var=0.003)
fedge = random_noise(fedgeOrig, mode='gaussian', mean=0.0, var=0.003)
fcr = random_noise(fcrOrig, mode='gaussian', mean=0.0, var=0.003)

# Select areas (MATLAB 1-based inclusive: 285:315)
xlow, xhigh = 285, 315
ylow, yhigh = 285, 315
fconst = fconst[xlow - 1:xhigh, ylow - 1:yhigh]
fedge = fedge[xlow - 1:xhigh, ylow - 1:yhigh]
fcr = fcr[xlow - 1:xhigh, ylow - 1:yhigh]

# Filter kernels
wy = np.array([-1.0, 0.0, 1.0], dtype=float).reshape(1, 3)
wx = wy.T

# Derivatives (MATLAB imfilter default correlation, zero padding)
dxconst = correlate(fconst, wx, mode='constant', cval=0.0)
dxedge = correlate(fedge, wx, mode='constant', cval=0.0)
dxcr = correlate(fcr, wx, mode='constant', cval=0.0)

dyconst = correlate(fconst, wy, mode='constant', cval=0.0)
dyedge = correlate(fedge, wy, mode='constant', cval=0.0)
dycr = correlate(fcr, wy, mode='constant', cval=0.0)

# Strip borders: MATLAB 3:end-3 => Python 2:-3
dxconst = dxconst[2:-3, 2:-3]
dxedge = dxedge[2:-3, 2:-3]
dxcr = dxcr[2:-3, 2:-3]
dyconst = dyconst[2:-3, 2:-3]
dyedge = dyedge[2:-3, 2:-3]
dycr = dycr[2:-3, 2:-3]

# Convert to vectors
dxconstv = dxconst.ravel()
dxedgev = dxedge.ravel()
dxcrv = dxcr.ravel()
dyconstv = dyconst.ravel()
dyedgev = dyedge.ravel()
dycrv = dycr.ravel()

# Display
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.ravel()

ax[0].imshow(fconstOrig, cmap='gray', vmin=0, vmax=1)
ax[0].set_title('No line')
ax[0].axis('off')

ax[1].imshow(fedgeOrig, cmap='gray', vmin=0, vmax=1)
ax[1].set_title('One line')
ax[1].axis('off')

ax[2].imshow(fcrOrig, cmap='gray', vmin=0, vmax=1)
ax[2].set_title('Two lines')
ax[2].axis('off')

ax[3].plot(dyconstv, dxconstv, 'k.', markersize=2)
ax[3].set_xlabel('f_{xx}')
ax[3].set_ylabel('f_{yy}')
ax[3].axis([-1, 1, -1, 1])

ax[4].plot(dyedgev, dxedgev, 'k.', markersize=2)
ax[4].set_xlabel('f_{xx}')
ax[4].set_ylabel('f_{yy}')
ax[4].axis([-1, 1, -1, 1])

ax[5].plot(dycrv, dxcrv, 'k.', markersize=2)
ax[5].set_xlabel('f_{xx}')
ax[5].set_ylabel('f_{yy}')
ax[5].axis([-1, 1, -1, 1])

fig.tight_layout()
fig.savefig('Figure1246.png')

print('Saved Figure1246.png')
plt.show()
