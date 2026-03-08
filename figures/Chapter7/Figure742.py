import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

# %% Data
RGB = img_as_float(imread(dip_data('jupiter-moon-closeup.tif')))
R = RGB[:, :, 0]
G = RGB[:, :, 1]
B = RGB[:, :, 2]

# %% Crop (MATLAB rect = [x, y, width, height], inclusive span)
x, y, w, h = 60, 240, 37, 80
RGBCrop = RGB[y:y + h + 1, x:x + w + 1, :]
RCrop = RGBCrop[:, :, 0]
GCrop = RGBCrop[:, :, 1]
BCrop = RGBCrop[:, :, 2]

# %% Segmentation
mean_r = np.mean(RCrop)
std_r = np.std(RCrop, ddof=1)
mean_g = np.mean(GCrop)
std_g = np.std(GCrop, ddof=1)
mean_b = np.mean(BCrop)
std_b = np.std(BCrop, ddof=1)

k = 1.25
X = (
    (R > mean_r - k * std_r) & (R < mean_r + k * std_r) &
    (G > mean_g - k * std_g) & (G < mean_g + k * std_g) &
    (B > mean_b - k * std_b) & (B < mean_b + k * std_b)
)

# %% Display
plt.figure(figsize=(9, 7))

plt.subplot(2, 2, 1)
plt.imshow(RGB)
plt.axis('off')
plt.title('RGB')

plt.subplot(2, 2, 2)
plt.imshow(RGBCrop)
plt.axis('off')
plt.title('RGB cropped')

plt.subplot(2, 2, 3)
plt.imshow(X, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title('Segmentation')

plt.tight_layout()
plt.savefig('Figure742.png')
plt.show()
