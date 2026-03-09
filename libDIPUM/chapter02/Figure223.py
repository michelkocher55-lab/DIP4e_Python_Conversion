import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data("Chronometer.tif")
f = imread(img_name)

# Convert to standard range if needed (skimage handles types well usually)
# resize returns float [0, 1] by default unless preserve_range is True.
# MATLAB imresize with 'nearest' usually keeps type or mimics it.

original_shape = f.shape[:2]

# Dimensions from MATLAB code:
# gnn1 = imresize(f, [689 690], 'nearest'); (300 dpi)
# gnn2 = imresize(f, [345 345], 'nearest'); (150 dpi)
# gnn3 = imresize(f, [165 166], 'nearest'); (72 dpi)

# Note: skimage resize takes shape as (rows, cols)

print("Reducing to 300 dpi...")
gnn1 = resize(f, (689, 690), order=0, preserve_range=True, anti_aliasing=False)

print("Reducing to 150 dpi...")
gnn2 = resize(f, (345, 345), order=0, preserve_range=True, anti_aliasing=False)

print("Reducing to 72 dpi...")
gnn3 = resize(f, (165, 166), order=0, preserve_range=True, anti_aliasing=False)

# Zoom back to original size
# fr300dpi = imresize(gnn1, size(f), 'nearest');
print("Resizing back to original...")
fr300dpi = resize(
    gnn1, original_shape, order=0, preserve_range=True, anti_aliasing=False
)
fr150dpi = resize(
    gnn2, original_shape, order=0, preserve_range=True, anti_aliasing=False
)
fr72dpi = resize(
    gnn3, original_shape, order=0, preserve_range=True, anti_aliasing=False
)

# Ensure displayable
# If preserve_range=True, values are in original range (e.g. 0-255).
# Matplotlib handles this if we don't normalize manually, but usually safe to cast or normalize.

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(fr300dpi.astype("uint8"), cmap="gray")
axes[1].set_title("Resol: 300 dpi (Nearest)")
axes[1].axis("off")

axes[2].imshow(fr150dpi.astype("uint8"), cmap="gray")
axes[2].set_title("Resol: 150 dpi (Nearest)")
axes[2].axis("off")

axes[3].imshow(fr72dpi.astype("uint8"), cmap="gray")
axes[3].set_title("Resol: 72 dpi (Nearest)")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("Figure223.png")
print("Saved Figure223.png")
plt.show()
