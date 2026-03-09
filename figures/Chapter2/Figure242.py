import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import AffineTransform, warp, estimate_transform
from libDIPUM.data_path import dip_data

# Data
img_name = dip_data("characterTestPattern688.tif")
f_orig = imread(img_name)
if f_orig.ndim == 3:
    f_orig = f_orig[:, :, 0]
f = img_as_float(f_orig)

h, w = f.shape

# Generate sheared image `gd`
# MATLAB: shear = [1 .4 0; .05 1 0; 0 0 1];
# This matrix is in MATLAB's post-multiply convention for [x y 1].
# x' = x + 0.05y
# y' = 0.4x + y
# Skimage AffineTransform matrix M applies to [x; y; 1] (pre-multiply).
# [x'; y'; 1] = M * [x; y; 1]
# x' = 1*x + 0.05*y + 0
# y' = 0.4*x + 1*y + 0
# So M = [[1, 0.05, 0], [0.4, 1, 0], [0, 0, 1]]

shear_matrix = np.array([[1, 0.05, 0], [0.4, 1, 0], [0, 0, 1]])

# Calculate output bounding box to simulate imtransform 'fully contained' behavior
corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])

# Transform corners
# AffineTransform operates on (x,y)
tform_shear = AffineTransform(matrix=shear_matrix)
corners_transformed = tform_shear(corners)

min_x = corners_transformed[:, 0].min()
max_x = corners_transformed[:, 0].max()
min_y = corners_transformed[:, 1].min()
max_y = corners_transformed[:, 1].max()

out_w = int(np.ceil(max_x - min_x))
out_h = int(np.ceil(max_y - min_y))

# We want output image (0..out_w, 0..out_h) to map to the transformed coords.
# Specifically, output pixel (u,v) should correspond to world coord (u + min_x, v + min_y).
# warp needs inverse_map: output_coord -> input_coord.
# input_coord = inverse_shear(output_coord_in_shear_space)
#             = inverse_shear(pixel_u + min_x, pixel_v + min_y)
#             = inverse_shear(pixel + offset)
# So effective transform for warp is Translation(offset) + InverseShear ??
# Actually, easiest is to compose transforms.
# We want T_warp such that T_warp(pixel) = input_pixel.
# pixel -> (add min) -> sheared_coord -> (inverse shear) -> input_pixel.

translation_matrix = np.array([[1, 0, min_x], [0, 1, min_y], [0, 0, 1]])
# Total M for [x'; y'; 1] = T * [x; y; 1]
# We want x_shear = x_pixel + min_x.
# This is T_trans * x_pixel.

# So forward map from Pixel -> Sheared is Translation(min_x, min_y).
# Then Sheared -> Input is Shear.inverse.
# So Pixel -> Input = Shear.inverse * Translation.

t_trans = AffineTransform(translation=(min_x, min_y))
# Combine: apply translation first, then inverse shear?
# logical chain: p_out --(trans)--> p_shear --(inv_shear)--> p_in.

# Skimage warp takes inverse_map.
# If we pass a transform, it assumes it maps output->input?
# Docs: "Inverse map... transforms coordinates in the output image into... input image".
# So yes, we construct the total transform T = Shear_inv * Translation.
# Note: Matrix multiplication order. M2 * M1 * v applies M1 then M2.
# We want Translation then InverseShear.
# So Matrix = M_inv_shear @ M_trans.

m_inv_shear = np.linalg.inv(shear_matrix)
m_trans = np.array([[1, 0, min_x], [0, 1, min_y], [0, 0, 1]])

map_matrix = m_inv_shear @ m_trans
tform_warp = AffineTransform(matrix=map_matrix)

gd = warp(f, tform_warp, output_shape=(out_h, out_w), order=1)

# Compute registration
# Control points (x, y)
base_points = np.array(
    [
        [114.9692, 109.4923],
        [618.2154, 75.7538],
        [600.3538, 610.9385],
        [75.0923, 633.4308],
    ]
)

input_points = np.array(
    [
        [118.5971, 155.0846],
        [623.2462, 322.6837],
        [629.7279, 851.9587],
        [104.7077, 663.9885],
    ]
)

# Estimate transform mapping input (gd) -> base (f)
tform_reg = estimate_transform("affine", src=input_points, dst=base_points)

# Back transformation
# We want to warp `gd` to match `f`.
# Target shape is f.shape.
# We iterate over `f` pixels (base).
# We need corresponding source pixels in `gd` (input).
# Source = Inverse(Transform(Input->Base)) * Dest ?
# Wait. `tform_reg` maps Input -> Base.
# We want Base -> Input (to sample from gd).
# So we need Inverse(tform_reg).
# warp(gd, inverse_map=tform_reg.inverse, output_shape=f.shape)

g = warp(gd, tform_reg.inverse, output_shape=f.shape, order=1)

# Difference
dif = f - g
# abs difference usually for display
dif_abs = np.abs(dif)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original Image f")
axes[0, 0].axis("off")

axes[0, 1].imshow(gd, cmap="gray")
axes[0, 1].set_title("Geometrically Distorted gd")
axes[0, 1].axis("off")

axes[1, 0].imshow(g, cmap="gray")
axes[1, 0].set_title("Recovered Image g")
axes[1, 0].axis("off")

axes[1, 1].imshow(dif_abs, cmap="gray")
axes[1, 1].set_title("Difference |f - g|")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure242.png")
print("Saved Figure242.png")
plt.show()
