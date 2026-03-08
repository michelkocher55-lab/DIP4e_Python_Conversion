import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIP.rgb2hsi4e import rgb2hsi4e
from libDIPUM.data_path import dip_data

# %% Data
RGB_u8 = imread(dip_data('jupiter-moon-closeup.tif'))
RGB = img_as_float(RGB_u8)

# %% Transform to HSI
HSI = rgb2hsi4e(RGB)
H = HSI[:, :, 0]
S = HSI[:, :, 1]
I = HSI[:, :, 2]

# %% Mask
T = 0.1 * np.max(S)
Mask = S > T

# %% Masked Hue
MaskedHue = Mask * H
Hist, edges = np.histogram(MaskedHue.ravel(), bins=256)
Bin = 0.5 * (edges[:-1] + edges[1:])

# %% Segmentation
X = MaskedHue > 0.9

# %% Display
plt.figure(figsize=(10, 14))

plt.subplot(4, 2, 1)
plt.imshow(RGB)
plt.axis('off')
plt.title('RGB')

plt.subplot(4, 2, 2)
plt.imshow(H, cmap='gray', vmin=np.min(H), vmax=np.max(H))
plt.axis('off')
plt.title('H')

plt.subplot(4, 2, 3)
plt.imshow(S, cmap='gray', vmin=np.min(S), vmax=np.max(S))
plt.axis('off')
plt.title('S')

plt.subplot(4, 2, 4)
plt.imshow(I, cmap='gray', vmin=np.min(I), vmax=np.max(I))
plt.axis('off')
plt.title('I')

plt.subplot(4, 2, 5)
plt.imshow(Mask, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title(f'Mask = S > {T:.4f}')

plt.subplot(4, 2, 6)
plt.imshow(MaskedHue, cmap='gray', vmin=np.min(MaskedHue), vmax=np.max(MaskedHue))
plt.axis('off')
plt.title('Masked Hue')

plt.subplot(4, 2, 7)
plt.bar(Bin, Hist, width=(Bin[1] - Bin[0]) if len(Bin) > 1 else 1.0)
plt.title('Hist (Masked Hue)')
plt.gca().set_box_aspect(1)
plt.axis('tight')

plt.subplot(4, 2, 8)
plt.imshow(X, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('Segmentation')

plt.tight_layout()
plt.savefig('Figure740.png')
plt.show()