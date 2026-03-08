"""Figure 13.24 - Perceptron example for patterns in Fig. 13.22."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from libDIP.perceptronTraining4e import perceptronTraining4e


print("Running Figure1324 (perceptron example)...")

# Data: augmented patterns (columns)
X = np.array([[1, 3], [1, 3], [1, 1]], dtype=float)
r = np.array([-1, 1], dtype=float)
w0 = np.array([0, 0, 0], dtype=float)

# Learning
w, epochs = perceptronTraining4e(X, r, alpha=1.0, nepochs=100, w0=w0)
print(f"Converged in {epochs} epochs to w = {w}")

# Decision surface shown intersecting the xy-plane.
# MATLAB: [x2, x1] = meshgrid(0:.1:4, 0:.1:4)
x2, x1 = np.meshgrid(np.arange(0, 4.0 + 0.1, 0.1), np.arange(0, 4.0 + 0.1, 0.1))

# Decision function d(x,y) = w1*x + w2*y + w3
d = w[0] * x1 + w[1] * x2 + w[2]
print(f"Decision boundary: {w[0]:.3f}*x + {w[1]:.3f}*y + {w[2]:.3f} = 0")

# Display
fig = plt.figure(1, figsize=(11, 5))

# Subplot 1: points + implicit decision boundary
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(X[0, 0], X[1, 0], "or")
ax1.plot(X[0, 1], X[1, 1], "og")

xx = np.linspace(0, 4, 400)
if abs(w[1]) > 1e-12:
    yy = -(w[0] * xx + w[2]) / w[1]
    ax1.plot(xx, yy, "k-")
elif abs(w[0]) > 1e-12:
    x_const = -w[2] / w[0]
    ax1.plot([x_const, x_const], [0, 4], "k-")

ax1.set_xlim(0, 4)
ax1.set_ylim(0, 4)
ax1.set_aspect("equal", adjustable="box")

# Subplot 2: decision surface + constant plane
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

# Use the same physical coordinate domain as subplot 1 for a clearer view.
surf_color = np.ones((41, 41, 3), dtype=float) * 0.75
ax2.plot_surface(x1, x2, d, facecolors=surf_color, linewidth=0, antialiased=True, shade=False, alpha=0.7)

C = 0.205 * float(np.max(d))
plane = np.zeros_like(x1, dtype=float) + C
ax2.plot_surface(x1, x2, plane, color="#9ecae1", linewidth=0, antialiased=True, shade=False, alpha=0.95)

ax2.view_init(elev=28, azim=-55)
ax2.grid(False)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_box_aspect((1, 1, 0.65))

# Save
out_path = os.path.join(os.path.dirname(__file__), "Figure1324.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
