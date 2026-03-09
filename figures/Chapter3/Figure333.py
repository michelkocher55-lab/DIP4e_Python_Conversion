import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.locstats import locstats
from libDIPUM.data_path import dip_data

# Data Loading

img_path = dip_data("hidden-symbols.tif")
f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]
f = img_as_float(f)

# Process
# param = [22.8, 0, 0.1, 0, 0.1];
# [g, GMF, GSTDF] = locstats(f, 3, 3, param);
# Note: locstats implementation might expect params as list or separate args?
# Usually MATLAB 'param' vector implies a single list/array argument.
# Checking if locstats.py follows this. Most transcoded funcs do.

param = [22.8, 0, 0.1, 0, 0.1]

g, GMF, GSTDF = locstats(f, 3, 3, param)

# Display
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(g, cmap="gray")
axes[1].set_title("Locally Enhanced Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure333.png")
print("Saved Figure333.png")
plt.show()
