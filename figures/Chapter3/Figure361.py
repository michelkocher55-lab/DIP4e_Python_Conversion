import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate

# Allow imports from project root (General, libDIP, libDIPUM)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from libDIPUM.zoneplate import zoneplate
from General.fir1 import fir1
from General.ftrans2 import ftrans2

# Data
f = zoneplate(8.2, 0.0275, 0)  # ~597 x 597

# Gaussian prefilter
w = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0
f = correlate(f, w, mode="nearest")  # MATLAB 'replicate'

# Lowpass filtering
lp, _ = fir1(128, 0.1)

# 2-D separable lowpass filter
lp2s = np.outer(lp, lp)

# 2-D circularly symmetric lowpass filter
lp2c = ftrans2(lp)

# Filter image with both filters (MATLAB 'symmetric')
glps = correlate(f, lp2s, mode="reflect")
glpc = correlate(f, lp2c, mode="reflect")

# Display
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(glps, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(glpc, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.tight_layout()
plt.savefig("Figure361.png")
print("Saved Figure361.png")
plt.show()
