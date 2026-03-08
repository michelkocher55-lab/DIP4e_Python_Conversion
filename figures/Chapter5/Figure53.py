import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

# %% Figure 5.3

# Data
f = imread(dip_data('test-pattern.tif'))
if f.ndim == 3:
    f = f[:, :, 0]

# MATLAB: f = im2double(f)
f = img_as_float(f)

# MATLAB: f = f(1:2:end, 1:2:end)
f = f[0::2, 0::2]
M, N = f.shape
print(f'Size after subsampling: M={M}, N={N}')

# Display
plt.figure(figsize=(6, 6))
plt.imshow(f, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.tight_layout()

# Print to file
plt.savefig('Figure53.png')
print('Saved Figure53.png')
plt.show()
