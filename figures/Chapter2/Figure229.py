
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.averaging4noisereduction import averaging4noisereduction
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

img_path = dip_data('sombrero-galaxy-original.tif')
forig = img_as_float(imread(img_path))

# Parameters from MATLAB code
# averaging4noisereduction(forig, 1, 'gaussian', 0, 64)
# Note: b=64 is extremely high for [0,1] image (std dev 64).
# DIPUM usually assumes parameters are in the same scale as image?
# Or maybe this example relies on intScaling4e to normalize the result (which is essentially pure noise + trace signal) back to visibility.
# We will trust the parameters.

# Noisy data (1 sample)
print("Generating noisy image results...")
# forig_noisy = intScaling4e(averaging4noisereduction(forig, 1, 'gaussian', 0, 64));
noisy_pure = averaging4noisereduction(forig, 1, 'gaussian', 0, 64)
forig_noisy = intScaling4e(noisy_pure)

# Denoising
# averages = [10, 50, 100, 500, 1000]
# We'll compute them.
# Note: This might take a moment.

ks = [10, 50, 100, 500, 1000]
results = []

for k in ks:
    print(f"Averaging {k} images...")
    res = averaging4noisereduction(forig, k, 'gaussian', 0, 64)
    scaled_res = intScaling4e(res)
    results.append(scaled_res)

fav10, fav50, fav100, fav500, fav1000 = results

# Display Results


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(forig_noisy, cmap='gray')
axes[0].set_title('Noisy Image (K=1)')
axes[0].axis('off')

axes[1].imshow(fav10, cmap='gray')
axes[1].set_title('Average of 10 Images')
axes[1].axis('off')

axes[2].imshow(fav50, cmap='gray')
axes[2].set_title('Average of 50 Images')
axes[2].axis('off')

axes[3].imshow(fav100, cmap='gray')
axes[3].set_title('Average of 100 Images')
axes[3].axis('off')

axes[4].imshow(fav500, cmap='gray')
axes[4].set_title('Average of 500 Images')
axes[4].axis('off')

axes[5].imshow(fav1000, cmap='gray')
axes[5].set_title('Average of 1000 Images')
axes[5].axis('off')

plt.tight_layout()
plt.savefig('Figure229.png')
print("Saved Figure229.png")
plt.show()