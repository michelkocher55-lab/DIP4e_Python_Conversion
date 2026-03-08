
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import correlate, uniform_filter
from libDIPUM.data_path import dip_data

print("Running Figure363 (Skeleton Bone Scan Enhancement)...")

# Image loading
img_path = dip_data('bonescan.tif')
f = imread(img_path)
if f.ndim == 3: f = f[:,:,0]

f = img_as_float(f)

# Laplacian
# w = [-1 -1 -1;-1 8 -1;-1 -1 -1];
w = np.array([[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]])

# gL = imfilter(f, w, 'symmetric');
gL = correlate(f, w, mode='reflect')

# Image needs scaling for display (gLs)
# Replicate intScaling4e behavior: map min..max to 0..1 (or full range)
gLs = gL.copy()
gL_min, gL_max = gLs.min(), gLs.max()
if gL_max > gL_min:
    gLs = (gLs - gL_min) / (gL_max - gL_min)

# Sharpen
# gSharp = f + gL;
gSharp = f + gL
# Clip to verify valid range?
# Usually we don't clip intermediate results unless display.
# But later calculation uses gSharp.

# Sobel Gradient
# gx = [-1 -2 -1;0 0 0;1 2 1];
# gy = [-1 0 1;-2 0 2;-1 0 1];
gx = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])
gy = np.array([[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]])

# G = abs(imfilter(f, gx, 'symmetric')) + abs(imfilter(f, gy, 'symmetric'));
Gx = correlate(f, gx, mode='reflect')
Gy = correlate(f, gy, mode='reflect')
G = np.abs(Gx) + np.abs(Gy)

# Smooth gradient
# waverage = ones(5)/25
waverage = np.ones((5, 5)) / 25.0
Gaverage = correlate(G, waverage, mode='reflect')

# Product
# LG = gSharp.*Gaverage;
LG = gSharp * Gaverage

# Edge Enhanced
# gEE = f + LG;
gEE = f + LG

# Display
# Two figures in MATLAB script. We can combine into one large figure or save as two files.
# The script saves Figure363.png and Figure363Bis.png

# Figure 1
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
axes1 = axes1.flatten()

axes1[0].imshow(f, cmap='gray')
axes1[0].set_title('Original')
axes1[0].axis('off')

axes1[1].imshow(gLs, cmap='gray')
axes1[1].set_title('Scaled Laplacian')
axes1[1].axis('off')

axes1[2].imshow(gSharp, cmap='gray') # Might need clipping for display
axes1[2].set_title('Image + Laplacian')
axes1[2].axis('off')

axes1[3].imshow(G, cmap='gray')
axes1[3].set_title('Sobel Gradient')
axes1[3].axis('off')

plt.tight_layout()
plt.savefig('Figure363.png')
print("Saved Figure363.png")

# Figure 2
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
axes2 = axes2.flatten()

axes2[0].imshow(Gaverage, cmap='gray')
axes2[0].set_title('Smoothed Sobel Gradient')
axes2[0].axis('off')

axes2[1].imshow(LG, cmap='gray') # Might need clipping
axes2[1].set_title('ProductMask * Sharpened')
axes2[1].axis('off')

axes2[2].imshow(gEE, cmap='gray') # Might need clipping
axes2[2].set_title('Result (f + LG)')
axes2[2].axis('off')

# 4th subplot was gamma commented out
axes2[3].axis('off')

plt.tight_layout()
plt.savefig('Figure363Bis.png')
print("Saved Figure363Bis.png")

plt.show()