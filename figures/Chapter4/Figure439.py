import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from libDIP.lpFilterTF4e import lpFilterTF4e

# Generate transfer function
meshILPF = lpFilterTF4e("ideal", 40, 40, 6)

# Display
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, projection="3d")

# MATLAB mesh: wireframe only
X, Y = np.meshgrid(np.arange(meshILPF.shape[1]), np.arange(meshILPF.shape[0]))
ax.plot_wireframe(X, Y, meshILPF, color="black", linewidth=0.4)
ax.set_axis_off()

plt.tight_layout()
plt.savefig("Figure439.png")
plt.show()
