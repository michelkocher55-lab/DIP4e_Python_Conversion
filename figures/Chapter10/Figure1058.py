import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import uniform_filter
from skimage.transform import resize

# Add project root so this script runs directly from any working directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from libDIPUM.nCutSegmentation import nCutSegmentation
from libDIPUM.mat2gray import mat2gray
from libDIPUM.data_path import dip_data


# Data
image_path = dip_data('building-600by600.tif')
f_raw = imread(image_path)

# Ensure grayscale
if f_raw.ndim == 3:
    f = f_raw.mean(axis=2)
else:
    f = f_raw.astype(float)

f = mat2gray(f)

# Smoothing
print('Applying smoothing (25x25)...')
I_smooth = uniform_filter(f, size=25, mode='nearest')

# Segment as in MATLAB example (half scale).
print('Running nCutSegmentation (sf=0.5)...')
S = nCutSegmentation(I_smooth, 2, sf=0.5)

# Force a clean binary map (0/1) with foreground in white.
labels = np.unique(S)
if labels.size != 2:
    raise RuntimeError(f'Expected 2 regions, got {labels.size}')

# Use a point near the bottom-center as foreground cue (building/hill region).
r0 = int(0.85 * S.shape[0])
c0 = int(0.50 * S.shape[1])
fg_label = S[r0, c0]
S_low_bin = (S == fg_label)

# Hybrid refinement:
# Keep robust low-res cut, but recover sharper 600x600 boundary by
# transferring class intensity statistics to full-resolution smoothed image.
I_low = resize(I_smooth, S.shape, order=1, preserve_range=True, anti_aliasing=True)
fg_mean = float(I_low[S_low_bin].mean())
bg_mean = float(I_low[~S_low_bin].mean())
t = 0.5 * (fg_mean + bg_mean)

if fg_mean >= bg_mean:
    Shalf = (I_smooth >= t).astype(float)
else:
    Shalf = (I_smooth < t).astype(float)

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
axes[0].imshow(f, cmap='gray')
axes[0].set_title('Original (600x600)')
axes[0].axis('off')

axes[1].imshow(I_smooth, cmap='gray')
axes[1].set_title('Smoothed (25x25 box)')
axes[1].axis('off')

axes[2].imshow(Shalf, cmap='gray', vmin=0, vmax=1)
axes[2].set_title('Graph Cut (2 regions)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure1058.png')
print('Saved Figure1058.png')
plt.show()
