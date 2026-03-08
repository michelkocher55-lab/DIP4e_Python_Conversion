import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.motion import motion
from libDIPUM.data_path import dip_data

# Parameters
MacroBlock = [16, 8, 8, 8]
Delta = np.array([[16, 16],
                  [8, 8],
                  [8, 8],
                  [8, 8]], dtype=int)
SubPixel = [1, 1, 0.5, 0.25]


def _invert_if_needed(img):
    # Match MATLAB visual polarity for these NASA TIFF frames.
    if np.issubdtype(img.dtype, np.integer):
        return np.iinfo(img.dtype).max - img
    return 1.0 - img


# Data
j = imread(dip_data('nasa67.tif'))
if j.ndim == 3:
    j = j[:, :, 0]
j = j[0:352, 118:470]
j = _invert_if_needed(j)

i = imread(dip_data('nasa79.tif'))
if i.ndim == 3:
    i = i[:, :, 0]
i = i[0:352, 118:470]
i = _invert_if_needed(i)

# Motion computation
# [e, a, dx, dy] = motion(i, j, MacroBlock(1), Delta(1, :), SubPixel(1));
e1, a1, dx1, dy1 = motion(i, j, MacroBlock[1], Delta[1, :], SubPixel[1])
e2, a2, dx2, dy2 = motion(i, j, MacroBlock[2], Delta[2, :], SubPixel[2])
e3, a3, dx3, dy3 = motion(i, j, MacroBlock[3], Delta[3, :], SubPixel[3])

# Difference (mat2gray(double(i)-double(j)))
d = i.astype(float) - j.astype(float)
dmin, dmax = np.min(d), np.max(d)
if dmax > dmin:
    dshow = (d - dmin) / (dmax - dmin)
else:
    dshow = np.zeros_like(d)

# Display
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(dshow, cmap='gray', vmin=0, vmax=1)
plt.title('Difference')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(e1, cmap='gray')  # MATLAB imshow(e1,[])
plt.title(f'e, m.block={MacroBlock[1]}, delta=[{Delta[1,0]} {Delta[1,1]}], SubP={SubPixel[1]}')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(e2, cmap='gray')
plt.title(f'e, m.block={MacroBlock[2]}, delta=[{Delta[2,0]} {Delta[2,1]}], SubP={SubPixel[2]}')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(e3, cmap='gray')
plt.title(f'e, m.block={MacroBlock[3]}, delta=[{Delta[3,0]} {Delta[3,1]}], SubP={SubPixel[3]}')
plt.axis('off')

plt.tight_layout()
plt.savefig('Figure835.png')
print('Saved Figure835.png')
plt.show()
