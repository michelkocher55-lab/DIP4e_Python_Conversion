import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import sys
from libDIPUM.otsuthresh import otsuthresh
from libDIPUM.data_path import dip_data

# Figure 10.36

# Data
img_path = dip_data('polymercell.tif')
I = imread(img_path)
if I.ndim == 3:
    I = I[:, :, 0]

# Compute histogram
if I.dtype == np.uint8:
    h, _ = np.histogram(I, bins=256, range=(0, 255))
else:
    h, _ = np.histogram(I, bins=256)

# Otsu threshold
T_norm, SM = otsuthresh(h)
T_otsu = T_norm * 255 if I.dtype == np.uint8 else T_norm
g_otsu = I > T_otsu

# Iterative global threshold (requested algorithm)
f = I.astype(float)
count = 0
T_iter = np.mean(f)
done = False

while not done:
    count += 1
    g = f > T_iter

    if np.any(g):
        m1 = np.mean(f[g])
    else:
        m1 = T_iter

    if np.any(~g):
        m2 = np.mean(f[~g])
    else:
        m2 = T_iter

    T_next = 0.5 * (m1 + m2)
    done = np.abs(T_iter - T_next) < 0.5
    T_iter = T_next

g_iter = f > T_iter

print(f"Otsu threshold: {T_otsu:.2f} (normalized {T_norm:.4f}), separability: {SM:.4f}")
print(f"Iterative threshold: {T_iter:.2f}, iterations: {count}")

# Display
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

axes[0].imshow(I, cmap='gray')
axes[0].set_title('Original Image (Polymersomes)')
axes[0].axis('off')

axes[1].plot(h, 'k')
line_x = T_otsu if I.dtype == np.uint8 else (T_otsu * 256)
axes[1].axvline(x=line_x, color='r', linestyle='--', label=f'Otsu T={T_otsu:.1f}')
axes[1].set_title('Histogram')
axes[1].legend()

axes[2].imshow(g_iter, cmap='gray')
axes[2].set_title(f'Iterative Mean (T={T_iter:.1f})')
axes[2].axis('off')

axes[3].imshow(g_otsu, cmap='gray')
axes[3].set_title(f'Otsu (SM={SM:.2f})')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('Figure1036.png', dpi=300, bbox_inches='tight')
print('Saved Figure1036.png')
plt.show()
