import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from libDIPUM.hpfilter import hpfilter

# IDEAL HIGHPASS
meshIHPF = np.fft.fftshift(hpfilter('ideal', 40, 40, 6))
IHPF = np.fft.fftshift(hpfilter('ideal', 600, 600, 100))

# GAUSSIAN HIGHPASS
meshGHPF = np.fft.fftshift(hpfilter('gaussian', 40, 40, 4))
GHPF = np.fft.fftshift(hpfilter('gaussian', 600, 600, 100))

# BUTTERWORTH HIGHPASS
meshBHPF = np.fft.fftshift(hpfilter('butterworth', 40, 40, 4, 2))
BHPF = np.fft.fftshift(hpfilter('butterworth', 600, 600, 100, 2))

# Display
fig = plt.figure(figsize=(12, 10))

# Mesh plots
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
X, Y = np.meshgrid(np.arange(meshIHPF.shape[1]), np.arange(meshIHPF.shape[0]))
ax1.plot_wireframe(X, Y, meshIHPF, color='black', linewidth=0.4)
ax1.set_axis_off()

ax4 = fig.add_subplot(3, 3, 4, projection='3d')
X, Y = np.meshgrid(np.arange(meshGHPF.shape[1]), np.arange(meshGHPF.shape[0]))
ax4.plot_wireframe(X, Y, meshGHPF, color='black', linewidth=0.4)
ax4.set_axis_off()

ax7 = fig.add_subplot(3, 3, 7, projection='3d')
X, Y = np.meshgrid(np.arange(meshBHPF.shape[1]), np.arange(meshBHPF.shape[0]))
ax7.plot_wireframe(X, Y, meshBHPF, color='black', linewidth=0.4)
ax7.set_axis_off()

# Images
ax2 = fig.add_subplot(3, 3, 2)
ax2.imshow(IHPF, cmap='gray')
ax2.axis('off')

ax5 = fig.add_subplot(3, 3, 5)
ax5.imshow(GHPF, cmap='gray')
ax5.axis('off')

ax8 = fig.add_subplot(3, 3, 8)
ax8.imshow(BHPF, cmap='gray')
ax8.axis('off')

# Profiles
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(IHPF[299, 300:])

ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(GHPF[299, 300:])

ax9 = fig.add_subplot(3, 3, 9)
ax9.plot(BHPF[299, 300:])

plt.tight_layout()
plt.savefig('Figure451.png')
plt.show()
