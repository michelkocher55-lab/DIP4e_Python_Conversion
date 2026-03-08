"""Figure 13.29 - Three activation functions."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


print("Running Figure1329 (activation functions)...")

# (a) Sigmoid
z = np.arange(-6.0, 6.0 + 0.01, 0.01)
hs = 1.0 / (1.0 + np.exp(-z))

# (b) tanh
ht = np.tanh(z)

# (c) ReLU
hReLU = np.maximum(0.0, z)

# Display
fig = plt.figure(1, figsize=(12, 4))

ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(hs, "k-")
ax1.set_xlim([0, 1200])
ax1.set_ylim([0, 1])
ax1.set_box_aspect(1)

ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(ht, "k-")
ax2.set_xlim([0, 1200])
ax2.set_ylim([-1, 1])
ax2.set_box_aspect(1)

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(hReLU, "k-")
ax3.set_xlim([0, 1200])
ax3.set_ylim([0, 6])
ax3.set_box_aspect(1)

out_path = os.path.join(os.path.dirname(__file__), "Figure1329.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
