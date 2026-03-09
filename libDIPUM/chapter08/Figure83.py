import numpy as np
import matplotlib.pyplot as plt

from fig81bc import fig81bc
from libDIP.histEqual4e import histEqual4e

# Figure 8.3

# Process
y = fig81bc("c")
z = histEqual4e(y)

# Display
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# MATLAB equivalent: plot(hist(double(y(:)), 1:256)), axis square
# For y in [0, 255], MATLAB hist with centers 1..256 maps value 0 to the first bin.
centers = np.arange(1, 257)
counts = np.bincount(y.astype(np.uint8).ravel(), minlength=256)
axes[0].plot(centers, counts)

axes[1].imshow(z, cmap="gray")

plt.tight_layout()

# Print to file
fig.savefig("Figure83.png", dpi=300, bbox_inches="tight")

plt.show()
