import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from libDIPUM.lpfilter import lpfilter

# Parameters
D0 = [10, 20, 40, 60]

# Generate transfer functions
BWLP = np.fft.fftshift(lpfilter("butterworth", 40, 40, 5, 2))
BWLPimage = np.fft.fftshift(lpfilter("butterworth", 1024, 1024, 128, 1))

# Profile extraction (improfile along x=301:600, y=300)
Profile = []
for d0 in D0:
    H = np.fft.fftshift(lpfilter("butterworth", 600, 600, d0))
    line = H[299, 300:600]
    Profile.append(line)
Profile = np.vstack(Profile).T

# Display
fig = plt.figure(figsize=(12, 4))

# Mesh
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
X, Y = np.meshgrid(np.arange(BWLP.shape[1]), np.arange(BWLP.shape[0]))
ax1.plot_wireframe(X, Y, BWLP, color="black", linewidth=0.4)
ax1.set_axis_off()
ax1.set_box_aspect((1, 1, 0.5))

# BWLP image
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(BWLPimage, cmap="gray")
ax2.axis("off")

# Profile plot
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(Profile)
# ax3.set_aspect('equal', adjustable='box')
ax3.legend([str(d0) for d0 in D0])

plt.tight_layout()
plt.savefig("Figure445.png")
plt.show()
