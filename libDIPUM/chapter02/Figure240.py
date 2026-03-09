import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import rotate
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

img_name = dip_data("letterT.tif")
f_orig = imread(img_name)
if f_orig.ndim == 3:
    f_orig = f_orig[:, :, 0]

# f = f(1:8:end, 1:8:end);
# Need lower resolution to show effects of rotation.
f = f_orig[::8, ::8]

# Convert to float for rotation logic consistency
# (skimage usually returns float)
f = img_as_float(f)

# Rotation
# MATLAB: imrotate(f, angle, method, bbox)
# angle in degrees, counter-clockwise. MATLAB uses negative for clockwise?
# MATLAB: + is CCW.
# Script uses -21.
# skimage rotate also uses degrees, + is CCW. So -21 matches.

# Methods:
# 'nearest' -> order=0
# 'bilinear' -> order=1
# 'bicubic' -> order=3

# 'crop' behavior:
# MATLAB 'crop': size of output is same as input.
# skimage resize=False means "size required to fit whole image" (MATLAB 'loose').
# resize=True means "resize to fit".
# Wait, skimage `rotate(resize=False)` pads to fit the rotated image (default).
# `resize=True` in skimage means "Change dimensions".
# Actually:
# skimage.transform.rotate(image, angle, resize=False, ...) -> Output shape is input shape?
# No, documentation says: "If True, output image size will be adjusted so that the entire rotated image fits."
# So `resize=False` keeps input shape (which matches MATLAB 'crop').

frn = rotate(f, -21, resize=False, order=0)
frbl = rotate(f, -21, resize=False, order=1)
frbc = rotate(f, -21, resize=False, order=3)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original (Subsampled)")
axes[0, 0].axis("off")

axes[0, 1].imshow(frn, cmap="gray")
axes[0, 1].set_title("Nearest Neighbor")
axes[0, 1].axis("off")

axes[1, 0].imshow(frbl, cmap="gray")
axes[1, 0].set_title("Bilinear")
axes[1, 0].axis("off")

axes[1, 1].imshow(frbc, cmap="gray")
axes[1, 1].set_title("Bicubic")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure240.png")
print("Saved Figure240.png")
plt.show()
