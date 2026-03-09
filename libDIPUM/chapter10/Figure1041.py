import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.gradlapthresh import gradlapthresh
from libDIPUM.data_path import dip_data

# Figure 10.41
# As in 10.40, but using the lower threshold.

# Data
f = imread(dip_data("yeast-cells.tif"))
if f.ndim == 3:
    f = f[:, :, 0]

# Use lower threshold (about 5% of maximum)
G = gradlapthresh(f, 1, 0.05)
print(G["G2"] * 255)  # Otsu threshold converted to [0,255] range
print(G["G3"])  # Percentile / separability output

# Display
plt.figure(figsize=(6, 6))
plt.imshow(G["G1"], cmap="gray")
plt.axis("off")

# Save to file
plt.tight_layout()
plt.savefig("Figure1041.png", dpi=300, bbox_inches="tight")
plt.show()
