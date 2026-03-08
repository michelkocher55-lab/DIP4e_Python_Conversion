
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte
from skimage.measure import label, regionprops
from libDIPUM.otsudualthresh import otsudualthresh
from libDIP.multithresh3E import multithresh3E
from libDIPUM.regiongrow import regiongrow
from libDIPUM.data_path import dip_data

def imhist(img):
    # Retrieve histogram 0-255
    if img.dtype != np.uint8:
        # Assuming [0,1] float or other. Check range.
        if img.max() <= 1.0:
            img = img_as_ubyte(img)
        else:
            img = img.astype(np.uint8)
            
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 255))
    return hist

def mat2gray(img):
    min_v = img.min()
    max_v = img.max()
    if max_v - min_v < 1e-10:
        return np.zeros_like(img)
    return (img - min_v) / (max_v - min_v)

def shrink_to_points(binary_mask):
    """
    Shrinks connected components to single pixels (like bwmorph shrink Inf).
    """
    lbl_img = label(binary_mask)
    props = regionprops(lbl_img)
    shrunk = np.zeros_like(binary_mask)
    for p in props:
        r, c = p.coords[0] # Pick first coordinate
        shrunk[r, c] = True
    return shrunk

# Data
image_path = dip_data('weldXray.tif')
f_raw = imread(image_path)

# Threshold seeds
Q = 254
S1 = f_raw > Q

# S = bwmorph(S1, 'shrink', Inf)
print("Shrinking seeds...")
S = shrink_to_points(S1)

# Difference image
f_double = img_as_float(f_raw)
diff_val = np.abs(((Q + 1) / 255.0) - f_double)
d = img_as_ubyte(diff_val)

# Histogram of difference
hd = imhist(d)
hd_norm = hd / np.sum(hd)

# Dual Otsu
print("Computing Otsu dual thresholds...")
T1, T2, _ = otsudualthresh(hd_norm)
print(f"T1={T1*255:.2f}, T2={T2*255:.2f}")

gtd = multithresh3E(d, [T1, T2])

gt1 = multithresh3E(d, [T1]) # Pass as list

# Region growing
print("Region growing...")
g, NR, SI, TI = regiongrow(f_double, S, T1)

# Display
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Row 1
axes[0, 0].imshow(f_raw, cmap='gray')
axes[0, 0].set_title('f')
axes[0, 0].axis('off')

axes[0, 1].plot(imhist(f_raw))
axes[0, 1].set_title('Hist(f)')
axes[0, 1].set_xlim([0, 255])

axes[0, 2].imshow(S1, cmap='gray')
axes[0, 2].set_title('S1 (Seeds)')
axes[0, 2].axis('off')

# Row 2
axes[1, 0].imshow(S, cmap='gray')
axes[1, 0].set_title('S (Shrunk Seeds)')
axes[1, 0].axis('off')

axes[1, 1].imshow(d, cmap='gray')
axes[1, 1].set_title('d (Diff Image)')
axes[1, 1].axis('off')

axes[1, 2].plot(hd_norm)
axes[1, 2].set_title('Hist(d) norm')
axes[1, 2].set_xlim([0, 255])

# Row 3
axes[2, 0].imshow(gtd, cmap='gray')
axes[2, 0].set_title('Dual Thresh (T1, T2)')
axes[2, 0].axis('off')

axes[2, 1].imshow(gt1, cmap='gray')
axes[2, 1].set_title('Thresh (T1)')
axes[2, 1].axis('off')

axes[2, 2].imshow(g, cmap='gray')
axes[2, 2].set_title('Region Growing Result')
axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig('Figure1046.png')
print("Saved Figure1046.png")
plt.show()