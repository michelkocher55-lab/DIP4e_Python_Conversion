import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from libDIPUM.bandfilter import bandfilter

# Parameters
r = 12

# Ideal
H = np.fft.fftshift(bandfilter('ideal', 'reject', 512, 512, 128, 60))
H1 = H[::r, ::r]

# Gaussian
H = np.fft.fftshift(bandfilter('gaussian', 'reject', 512, 512, 128, 60))
H2 = H[::r, ::r]

# Butterworth
H = np.fft.fftshift(bandfilter('butterworth', 'reject', 512, 512, 128, 60, 1))
H3 = H[::r, ::r]

# Display figure 1 (mesh)
fig1 = plt.figure(figsize=(12, 4))

ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
X, Y = np.meshgrid(np.arange(H1.shape[1]), np.arange(H1.shape[0]))
ax1.plot_wireframe(X, Y, H1, color='black', linewidth=0.4)
ax1.set_axis_off()
ax1.set_box_aspect((1, 1, 1))

ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
X, Y = np.meshgrid(np.arange(H2.shape[1]), np.arange(H2.shape[0]))
ax2.plot_wireframe(X, Y, H2, color='black', linewidth=0.4)
ax2.set_axis_off()
ax2.set_box_aspect((1, 1, 1))

ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
X, Y = np.meshgrid(np.arange(H3.shape[1]), np.arange(H3.shape[0]))
ax3.plot_wireframe(X, Y, H3, color='black', linewidth=0.4)
ax3.set_axis_off()
ax3.set_box_aspect((1, 1, 1))

plt.tight_layout()
plt.savefig('Figure462.png')

# Display figure 2 (images)
fig2, axes = plt.subplots(1, 3, figsize=(12, 4))

# autoscale like imshow(..., [])
def autoscale(img):
    img = np.asarray(img, dtype=float)
    img = img - img.min()
    maxv = img.max()
    if maxv > 0:
        img = img / maxv
    return img

axes[0].imshow(autoscale(H1), cmap='gray')
axes[0].axis('off')

axes[1].imshow(autoscale(H2), cmap='gray')
axes[1].axis('off')

axes[2].imshow(autoscale(H3), cmap='gray')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure463.png')
plt.show()
