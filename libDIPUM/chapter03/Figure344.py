import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate, uniform_filter
from libDIPUM.gaussiankernel import gaussiankernel

# Data: Synthetic image
f = np.zeros((1024, 1024), dtype=np.uint8)
f[127:896, 191:320] = 255

# Kernels
# box = ones(71); box = box/sum(box(:));
# We can use uniform_filter.

# Gaussian
# gaussian = gaussiankernel(151, 'sampled', 25, 1);
gaussian, _ = gaussiankernel(151, "sampled", 25.0, 1.0)
gaussian = gaussian / np.sum(gaussian)

# Filtering
# MATLAB: imfilter(f, box, 'replicate')
# scipy: uniform_filter or correlated. uniform_filter is faster for box.
# mode='nearest' corresponds to 'replicate'.

f_float = f.astype(float) / 255.0

# Box filtering
# uniform_filter size=71
gbox = uniform_filter(f_float, size=71, mode="nearest")

# Gaussian filtering
ggauss = correlate(f_float, gaussian, mode="nearest")

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(gbox, cmap="gray")
axes[1].set_title("Box Filter 71x71")
axes[1].axis("off")

axes[2].imshow(ggauss, cmap="gray")
axes[2].set_title("Gaussian Filter 151x151, sigma=25")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure344.png")
print("Saved Figure344.png")
plt.show()
