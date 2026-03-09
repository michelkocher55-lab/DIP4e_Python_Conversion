import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data("Chronometer.tif")
f = imread(img_name)

# Convert to standard range/type logic
# resize returns float [0, 1] if input is float or if we don't preserve range.

# Dimensions from MATLAB:
# Original is large (2136x2140 likely).
# Target 72 dpi size: [165, 166] (rows, cols)

target_shape = (165, 166)
original_shape = f.shape[:2]

# Nearest Neighbor (order=0)
print("Processing Nearest Neighbor...")
g72nn = resize(f, target_shape, order=0, preserve_range=True, anti_aliasing=False)
f72zoomnn = resize(
    g72nn, original_shape, order=0, preserve_range=True, anti_aliasing=False
)

# Bilinear (order=1)
print("Processing Bilinear...")
# Note: skimage resize anti_aliasing default is True for order > 0, usually good to keep False to match MATLAB strict imresize without anti-aliasing filter if not specified?
# MATLAB imresize default enables anti-aliasing for shrinking.
# But here we specify method. 'nearest' turns it off. 'bilinear'/'bicubic' usually have it on by default in MATLAB when shrinking?
# The MATLAB script doesn't explicitly say 'Antialiasing', false.
# However, for pure interpolation comparison, maybe we should let it smooth.
# Let's stick to default behavior for order>0 which usually implies some smoothing or just interpolation.
g72bl = resize(f, target_shape, order=1, preserve_range=True)
f72zoombl = resize(g72bl, original_shape, order=1, preserve_range=True)

# Bicubic (order=3)
print("Processing Bicubic...")
g72bc = resize(f, target_shape, order=3, preserve_range=True)
f72zoombc = resize(g72bc, original_shape, order=3, preserve_range=True)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

axes[0].imshow(f72zoomnn.astype("uint8"), cmap="gray")
axes[0].set_title("Nearest Neighbor")
axes[0].axis("off")

axes[1].imshow(f72zoombl.astype("uint8"), cmap="gray")
axes[1].set_title("Bilinear")
axes[1].axis("off")

axes[2].imshow(f72zoombc.astype("uint8"), cmap="gray")
axes[2].set_title("Bicubic")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure227.png")
print("Saved Figure227.png")

plt.show()
