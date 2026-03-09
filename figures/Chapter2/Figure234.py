import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.data_path import dip_data

# Load and convert to double (0-1 float)
f = img_as_float(imread(dip_data("dentalXray.tif")))
h = img_as_float(imread(dip_data("dentalXrayMask.tif")))

# Check sizes
if f.shape != h.shape:
    print(
        "Warning: Image and mask have different dimensions. Resizing mask might be needed."
    )
    # Proceeding assuming they match as per book figures usually.

# Masking (Multiplication)
# g = f.*h;
g = f * h

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Image (f)")
axes[0].axis("off")

axes[1].imshow(h, cmap="gray")
axes[1].set_title("Mask (h)")
axes[1].axis("off")

axes[2].imshow(g, cmap="gray")
axes[2].set_title("Product (g = f * h)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure234.png")
print("Saved Figure234.png")

plt.show()
