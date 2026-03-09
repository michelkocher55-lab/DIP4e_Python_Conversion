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
from helpers.fir1 import fir1
from helpers.ftrans2 import ftrans2
from libDIP.intScaling4e import intScaling4e

# Data
f = zoneplate(8.2, 0.0275, 0)  # ~597 x 597

# Lowpass filtering
lp, _ = fir1(128, 0.1)

# 2-D separable lowpass filter
lp2s = np.outer(lp, lp)

# 2-D circularly symmetric lowpass filter
lp2c = ftrans2(lp)

# Apply filters (MATLAB 'symmetric')
glps = correlate(f, lp2s, mode="reflect")
glpc = correlate(f, lp2c, mode="reflect")

# Highpass from lowpass
ghp = f - glpc

# Alternative highpass from impulse - lowpass
M = lp.size
center = int(np.ceil(M / 2.0)) - 1  # zero-based index
_delta = np.zeros(M, dtype=float)
_delta[center] = 1.0
hp = _delta - lp

hp2c = ftrans2(hp)
ghpc = correlate(f, hp2c, mode="reflect")

# Bandreject from lowpass/highpass
lp1, _ = fir1(128, 0.06)
lp2, _ = fir1(128, 0.12)
hp2 = _delta - lp2

hbr = lp1 + hp2
hbrc = ftrans2(hbr)
gbr = correlate(f, hbrc, mode="reflect")

# Bandpass from bandreject
hbp = _delta - hbr
hbpc = ftrans2(hbp)
gbp = correlate(f, hbpc, mode="reflect")

# Display
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(glpc, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(ghpc, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(intScaling4e(ghpc), cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gbr, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gbp, cmap="gray", vmin=0, vmax=1)
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(intScaling4e(gbp), cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("Figure362.png")
print("Saved Figure362.png")
plt.show()
