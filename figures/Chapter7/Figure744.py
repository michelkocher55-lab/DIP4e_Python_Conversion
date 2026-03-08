import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float

from libDIPUM.colorgrad import colorgrad
from libDIPUM.data_path import dip_data

# %% Data
RGB = img_as_float(imread(dip_data('lenna-RGB.tif')))
R = RGB[:, :, 0]
G = RGB[:, :, 1]
B = RGB[:, :, 2]
_ = (R, G, B)

# %% Gradient
VectorGradient, Angle, PerPlaneGradient = colorgrad(RGB)
Diff = VectorGradient - PerPlaneGradient
_ = Angle

# %% Display
plt.figure(figsize=(9, 7))

plt.subplot(2, 2, 1)
plt.imshow(RGB)
plt.axis('off')
plt.title('RGB')

plt.subplot(2, 2, 2)
plt.imshow(VectorGradient, cmap='gray', vmin=np.min(VectorGradient), vmax=np.max(VectorGradient))
plt.axis('off')
plt.title('Vector gradient')

plt.subplot(2, 2, 3)
plt.imshow(PerPlaneGradient, cmap='gray', vmin=np.min(PerPlaneGradient), vmax=np.max(PerPlaneGradient))
plt.axis('off')
plt.title('Per plane gradient')

plt.subplot(2, 2, 4)
plt.imshow(Diff, cmap='gray', vmin=np.min(Diff), vmax=np.max(Diff))
plt.axis('off')
plt.title('Difference')

plt.tight_layout()
plt.savefig('Figure744.png')
plt.show()
