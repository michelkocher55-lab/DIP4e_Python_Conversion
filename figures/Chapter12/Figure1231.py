import os
import matplotlib.pyplot as plt
from skimage.io import imread

print("Running Figure1231 (Three texture strips)...")

base_dir = "/Users/michelkocher/michel/Data/DIP-DIPUM/DIP"

# Data
f1 = imread(os.path.join(base_dir, "strip-uniform-noise.tif"))
f2 = imread(os.path.join(base_dir, "strip-2Dsinusoidal-waveform.tif"))
f3 = imread(os.path.join(base_dir, "strip-cktboard-section.tif"))

# Display
fig, ax = plt.subplots(3, 1, figsize=(8, 10))

ax[0].imshow(f1, cmap="gray")
ax[0].axis("off")

ax[1].imshow(f2, cmap="gray")
ax[1].axis("off")

ax[2].imshow(f3, cmap="gray")
ax[2].axis("off")

fig.tight_layout()
fig.savefig("Figure1231.png")

print("Saved Figure1231.png")
plt.show()
