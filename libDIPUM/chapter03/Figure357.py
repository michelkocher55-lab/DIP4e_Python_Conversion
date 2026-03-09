import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import correlate
from libDIPUM.data_path import dip_data

# Sobel kernels
wh = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
wv = wh.T

# Data
f = imread(dip_data("contact-lens.tif"))
if f.ndim == 3:
    f = f[:, :, 0]

# Filtering (MATLAB 'symmetric' -> scipy 'reflect')
gx = np.abs(correlate(f.astype(float), wv, mode="reflect"))
gy = np.abs(correlate(f.astype(float), wh, mode="reflect"))
g = gx + gy

# Display
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(f, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(g, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("Figure357.png")
print("Saved Figure357.png")
plt.show()
