import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Allow imports from project root (General, libDIP, libDIPUM)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from helpers.fir1 import fir1
from helpers.ftrans2 import ftrans2

# Lowpass filtering
lp, _ = fir1(128, 0.1)

# 2-D separable lowpass filter
lp2s = np.outer(lp, lp)

# 2-D Circularly symmetric lowpass filter
lp2c = ftrans2(lp)

# Display
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(lp, color="k")
ax1.set_box_aspect(1)  # MATLAB-like axis square (subplot box square)
ax1.margins(x=0)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
Z = lp2c[::2, ::2]
X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
ax2.plot_wireframe(X, Y, Z, color="k", linewidth=0.5)
ax2.set_axis_off()

plt.tight_layout()
plt.savefig("Figure360.png")
print("Saved Figure360.png")
plt.show()
