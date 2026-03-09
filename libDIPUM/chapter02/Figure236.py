import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("skeleton.tif")

# Load and convert to double (0-1 float)
A = img_as_float(imread(img_path))

# Negate (Complement)
# An = 1 - A
An = 1.0 - A

# Union (Max) with a constant constant image B
# MATLAB: B = 3*mean2(A)*ones; -> Scalar B = 3 * mean(A)
# union = max(A, B);

B_val = 3 * np.mean(A)
# Ensure B isn't > 1 if we want it to be valid gray level, but typically fuzzy sets are [0,1].
# If 3*mean(A) > 1, the max will be saturated at that value?
# Or just clipped for display?
# Let's perform the math.

# Note: numpy.maximum performs element-wise max.
Union = np.maximum(A, B_val)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

axes[0].imshow(A, cmap="gray")
axes[0].set_title("Original Image (A)")
axes[0].axis("off")

axes[1].imshow(An, cmap="gray")
axes[1].set_title("Complement (1 - A)")
axes[1].axis("off")

axes[2].imshow(Union, cmap="gray", vmin=0, vmax=1)
axes[2].set_title(f"Union (max(A, 3*mean(A))) \n3*mean(A)={B_val:.2f}")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure236.png")
print("Saved Figure236.png")

plt.show()
