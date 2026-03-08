
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_ubyte
from libDIPUM.im2bitplanes import im2bitplanes
from libDIPUM.bitplanes2im import bitplanes2im
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data('drip-bottle.tif')

f256 = imread(img_name)

# Get bit planes
B = im2bitplanes(f256, 8)

# Reduce image to 7 bits (planes 2-8 in MATLAB -> 1-7 in Python 0-based)
# MATLAB: [2 3 4 5 6 7 8]
# Python: [1, 2, 3, 4, 5, 6, 7]
f128 = img_as_ubyte(bitplanes2im(B, [1, 2, 3, 4, 5, 6, 7]))

# Repeat down to 2 bits
# MATLAB: f64 = ... [3 4 5 6 7 8] -> Py: [2, 3, 4, 5, 6, 7]
f64 = img_as_ubyte(bitplanes2im(B, range(2, 8))) # 2,3,4,5,6,7

# f32 = ... [4 5 6 7 8] -> Py: [3, 4, 5, 6, 7]
f32 = img_as_ubyte(bitplanes2im(B, range(3, 8)))

# f16 = ... [5 6 7 8] -> Py: [4, 5, 6, 7]
f16 = img_as_ubyte(bitplanes2im(B, range(4, 8)))

# f8 = ... [6 7 8] -> Py: [5, 6, 7]
f8 = img_as_ubyte(bitplanes2im(B, range(5, 8)))

# f4 = ... [7 8] -> Py: [6, 7]
f4 = img_as_ubyte(bitplanes2im(B, range(6, 8)))

# f2 = ... [8] -> Py: [7]
f2 = img_as_ubyte(bitplanes2im(B, [7]))

# Display Figure 1
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
axes1 = axes1.flatten()

axes1[0].imshow(f256, cmap='gray')
axes1[0].set_title('Original (8 bits)')
axes1[0].axis('off')

axes1[1].imshow(f128, cmap='gray')
axes1[1].set_title('7 bits')
axes1[1].axis('off')

axes1[2].imshow(f64, cmap='gray')
axes1[2].set_title('6 bits')
axes1[2].axis('off')

axes1[3].imshow(f32, cmap='gray')
axes1[3].set_title('5 bits')
axes1[3].axis('off')

plt.tight_layout()
plt.savefig('Figure224.png')
print("Saved Figure224.png")

# Display Figure 2
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
axes2 = axes2.flatten()

axes2[0].imshow(f16, cmap='gray')
axes2[0].set_title('4 bits')
axes2[0].axis('off')

axes2[1].imshow(f8, cmap='gray')
axes2[1].set_title('3 bits')
axes2[1].axis('off')

axes2[2].imshow(f4, cmap='gray')
axes2[2].set_title('2 bits')
axes2[2].axis('off')

axes2[3].imshow(f2, cmap='gray')
axes2[3].set_title('1 bit')
axes2[3].axis('off')

plt.tight_layout()
plt.savefig('Figure224Bis.png')
print("Saved Figure224Bis.png")

plt.show()