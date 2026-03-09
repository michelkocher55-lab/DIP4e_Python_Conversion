import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import invert
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data("Chronometer.tif")

f = imread(img_name)

# Negate (Complement)
# g = imcomplement(f)
g = invert(f)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()

# Use 'gray' cmap. If image is RGB, it handles it. If gray, it uses map.
# Chronometer.tif is likely grayscale? If RGB, invert still works element-wise.
axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(g, cmap="gray")
axes[1].set_title("Complement (Negative)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure238.png")
print("Saved Figure238.png")

plt.show()
