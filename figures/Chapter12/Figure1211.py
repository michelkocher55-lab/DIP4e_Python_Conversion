import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.bwboundaries import bwboundaries
from libDIPUM.signature import signature
from libDIPUM.data_path import dip_data

print("Running Figure1211 (Boundary Signatures)...")

# 1. Load Data
# Paths based on previous search
path_square = dip_data("binary-square-distorted.tif")
path_triangle = dip_data("binary-triangle-distorted.tif")
f1 = imread(path_square)
f2 = imread(path_triangle)

# 2. Boundaries
B1_list = bwboundaries(f1, conn=8)
B2_list = bwboundaries(f2, conn=8)

if len(B1_list) == 0 or len(B2_list) == 0:
    print("Error: No boundaries found.")


B1 = B1_list[0]
B2 = B2_list[0]

# 3. Signatures
dist1, angle1 = signature(B1)
dist2, angle2 = signature(B2)

# 4. Display
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(f1, cmap="gray")
axes[0, 0].set_title("B1 (Square Distorted)")
axes[0, 0].axis("off")

axes[0, 1].imshow(f2, cmap="gray")
axes[0, 1].set_title("B2 (Triangle Distorted)")
axes[0, 1].axis("off")

# Signatures
axes[1, 0].plot(angle1, dist1, "k-")
axes[1, 0].set_title(r"Signature $S_1(\theta)$")
axes[1, 0].set_xlabel("Angle (degrees)")
axes[1, 0].set_ylabel("Distance")
axes[1, 0].grid(True)
axes[1, 0].set_xlim([0, 360])

axes[1, 1].plot(angle2, dist2, "k-")
axes[1, 1].set_title(r"Signature $S_2(\theta)$")
axes[1, 1].set_xlabel("Angle (degrees)")
axes[1, 1].set_ylabel("Distance")
axes[1, 1].grid(True)
axes[1, 1].set_xlim([0, 360])

plt.tight_layout()

plt.savefig("Figure1211.png")
print("Saved Figure1211.png")
plt.show()
