import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

print("Running Figure1231 (Three texture strips)...")


# Data
f1 = imread(dip_data("strip-uniform-noise.tif"))
f2 = imread(dip_data("strip-2Dsinusoidal-waveform.tif"))
f3 = imread(dip_data("strip-cktboard-section.tif"))

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
