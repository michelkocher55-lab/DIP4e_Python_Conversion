import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.intensityTransformations import intensityTransformations
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("retina.tif")
f = imread(img_path)

# Simulate monitor gamma
gmonitor = intensityTransformations(f, "gamma", 2.5)

# Gamma correction with 1/2.5
ggammacorrected = intensityTransformations(f, "gamma", 0.4)

# Put corrected image through monitor
gmonitorcorrected = intensityTransformations(ggammacorrected, "gamma", 2.5)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].axis("off")

axes[0, 1].imshow(gmonitor, cmap="gray")
axes[0, 1].axis("off")

axes[1, 0].imshow(ggammacorrected, cmap="gray")
axes[1, 0].axis("off")

axes[1, 1].imshow(gmonitorcorrected, cmap="gray")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure37.png")
plt.show()
