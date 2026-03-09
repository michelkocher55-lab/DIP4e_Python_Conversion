import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from libDIPUM.lpfilter import lpfilter

# Parameters
D0 = [10, 20, 40, 60]

# Generate transfer functions
GLPF40 = np.fft.fftshift(lpfilter("gaussian", 600, 600, 40))
meshGLPF40 = np.fft.fftshift(lpfilter("gaussian", 40, 40, 4))

# Profile extraction (improfile along x=301:600, y=300)
Profile = []
for d0 in D0:
    H = np.fft.fftshift(lpfilter("gaussian", 600, 600, d0))
    # MATLAB indices 301:600 (1-based) -> Python 300:600 (0-based)
    line = H[299, 300:600]
    Profile.append(line)
Profile = np.vstack(Profile).T  # columns correspond to D0 values

# Display
fig = plt.figure(figsize=(12, 4))

# Mesh
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
X, Y = np.meshgrid(np.arange(meshGLPF40.shape[1]), np.arange(meshGLPF40.shape[0]))
ax1.plot_wireframe(X, Y, meshGLPF40, color="black", linewidth=0.4)
ax1.set_axis_off()
ax1.set_box_aspect((1, 1, 0.5))

# GLPF40 image
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(GLPF40, cmap="gray")
ax2.axis("off")

# Profile plot
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(Profile)
# ax3.set_aspect('equal', adjustable='box')
ax3.legend([str(d0) for d0 in D0])

plt.tight_layout()
plt.savefig("Figure443.png")
plt.show()
