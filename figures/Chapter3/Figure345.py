import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import correlate
from libDIPUM.gaussiankernel import gaussiankernel
from libDIPUM.data_path import dip_data

# Image loading
img_name = dip_data("testpattern1024.tif")
f = imread(img_name)
if f.ndim == 3:
    f = f[:, :, 0]

f = img_as_float(f)

# Kernel
# gauss187 = gaussiankernel(187, 'sampled', 31, 1);
gauss187, _ = gaussiankernel(187, "sampled", 31.0, 1.0)
gauss187 = gauss187 / np.sum(gauss187)

# Filtering with different boundary conditions

# 1. Zero padding
# MATLAB: imfilter(f, h) (default is zero)
# scipy: mode='constant', cval=0.0
gzeropad = correlate(f, gauss187, mode="constant", cval=0.0)

# 2. Symmetric padding
# MATLAB: imfilter(f, h, 'symmetric')
# MATLAB 'symmetric' pads with mirror reflections of itself.
# padarray([1 2 3], 2, 'symmetric') -> [2 1 1 2 3 3 2]. (Repeats edge pixel).
# scipy.ndimage.correlate mode='reflect' -> d c b a | a b c d | d c b a (Repeats edge pixel).
# mode='mirror' -> d c b | a b c d | c b a (Does not repeat edge pixel).
# So MATLAB 'symmetric' matches scipy 'reflect'.
gsymmpad = correlate(f, gauss187, mode="reflect")

# 3. Replicate padding
# MATLAB: imfilter(f, h, 'replicate')
# scipy: mode='nearest'
greplpad = correlate(f, gauss187, mode="nearest")

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(gzeropad, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Zero Padding")
axes[1].axis("off")

axes[2].imshow(gsymmpad, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Symmetric Padding")
axes[2].axis("off")

axes[3].imshow(greplpad, cmap="gray", vmin=0, vmax=1)
axes[3].set_title("Replicate Padding")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("Figure345.png")
print("Saved Figure345.png")
plt.show()
