"""Figure 13.25 - Quadratic functions for 1 and 2 variables."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


print("Running Figure1325 (quadratic functions)...")

# Plot of 1D quadratic function.
x = np.arange(0.0, 2.0 + 0.01, 0.01)
r = 1.0
E = 0.5 * ((r - x) ** 2)

# Plot of 2D quadratic function.
w2, w1 = np.meshgrid(
    np.arange(0.0, 2.0 + 0.005, 0.005), np.arange(0.0, 2.0 + 0.01, 0.01)
)
f = 0.5 * ((w1 - 1.0) ** 2 + (w2 - 1.0) ** 2)

# Display
fig = plt.figure(1, figsize=(10, 4.5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(E, "k-")
ax1.set_box_aspect(1)
ax1.autoscale(enable=True, axis="both", tight=True)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
row_idx = slice(0, f.shape[0], 8)
col_idx = slice(0, f.shape[1], 8)
X = np.arange(f[row_idx, col_idx].shape[1])
Y = np.arange(f[row_idx, col_idx].shape[0])
Xg, Yg = np.meshgrid(X, Y)
ax2.plot_wireframe(Xg, Yg, f[row_idx, col_idx], color="k", linewidth=0.7)
ax2.set_box_aspect((1, 1, 0.6))

out_path = os.path.join(os.path.dirname(__file__), "Figure1325.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
