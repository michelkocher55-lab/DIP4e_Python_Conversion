import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

# Image loading (Exact path)
img_name = dip_data("trophozoite.tif")
f = imread(img_name)
if f.ndim == 3:
    f = f[:, :, 0]

# Bit slicing
# MATLAB loop implementation:
# Res = f
# for i = 1:8
#    Bit(:,:,i) = Res >= 2^(NBits-i)
#    Res = Res - Bit * 2^(NBits-i)

# This extracts bits from MSB (128) down to LSB (1).
# i=1: 2^7=128. Bit plane 7 (MSB).
# i=8: 2^0=1.   Bit plane 0 (LSB).

NBits = 8

# We can do this efficiently with bitwise AND
# But replicating loop/order:

Bits = []
# Loop 1 to 8
# Using numpy bitwise is safer than manual subtraction logic for types,
# but manual subtraction works if types are uint8/int.

# Let's use standard bitwise extraction which is cleaner in Python
# But adhere to the order: MSB first (i=1 matches MSB).
for i in range(NBits):
    # i goes 0..7
    # Power of 2: NBits - 1 - i.
    # i=0 -> 7 (128). i=7 -> 0 (1).
    bit_pos = NBits - 1 - i

    # Extract bit
    # (f >> bit_pos) & 1
    b_img = (f >> bit_pos) & 1
    Bits.append(b_img)

# Display
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

# Plot 1: Original
axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis("off")

# Plot 2..9: Bit planes
# MATLAB: title(['b', num2str(i-1)]).
# i=1 (MSB) -> 'b0'. i=8 (LSB) -> 'b7'.
# This naming is arguably confusing (usually b7 is MSB), but I will replicate the MATLAB script Output.
# The MATLAB script uses loop i=1:8.
# i=1 (MSB). Title 'b0'.

for i in range(NBits):
    ax = axes[i + 1]
    ax.imshow(Bits[i], cmap="gray")
    ax.set_title(f"b{i}")  # i starts at 0. matches MATLAB 'i-1'.
    ax.axis("off")

plt.tight_layout()
plt.savefig("Figure314.png")
print("Saved Figure314.png")
plt.show()
