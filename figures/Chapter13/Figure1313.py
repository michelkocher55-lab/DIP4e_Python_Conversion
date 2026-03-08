
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import match_template
import ia870 as ia
from libDIPUM.data_path import dip_data

# Data
path_f = dip_data('Fig1209(a)(Hurricane Andrew).tif')
path_h = dip_data('Fig1209(b)(eye template).tif')
f = imread(path_f)
h = imread(path_h)

# Correlation
Corr = match_template(f, h)

# Find max
ij = np.unravel_index(np.argmax(Corr), Corr.shape)
RowMaxCorr, ColMaxCorr = ij[0], ij[1] # Top-left coordinates of the match

# MaxCorrImg
MaxCorrImg = np.zeros_like(f, dtype=bool)

# We can mark the center of the template for better visualization
h_h, h_w = h.shape
center_r = RowMaxCorr + h_h // 2
center_c = ColMaxCorr + h_w // 2

# Ensure bounds
center_r =  min(max(center_r, 0), f.shape[0]-1)
center_c = min(max(center_c, 0), f.shape[1]-1)

MaxCorrImg[center_r, center_c] = 1

# Display
fig = plt.figure(figsize=(10, 8))
fig.canvas.manager.set_window_title('Correlation')

# 1. Image f
plt.subplot(2, 2, 1)
plt.imshow(f, cmap='gray')
plt.title('f')
plt.axis('off')

# 2. Template h
plt.subplot(2, 2, 2)
plt.imshow(h, cmap='gray')
plt.title('h = Template')
plt.axis('off')

# 3. Correlation surface
plt.subplot(2, 2, 3)
plt.imshow(Corr, cmap='gray')
plt.title('r_{fh}')
plt.colorbar()
plt.axis('off')

# 4. Max dilated
marker_dilated = ia.iadil(MaxCorrImg, ia.iasecross(5))

plt.subplot(2, 2, 4)
plt.imshow(marker_dilated, cmap='gray')
plt.title('max(r_{fh})')
plt.axis('off')

plt.tight_layout()
plt.savefig('Figure1313.png')
plt.show()