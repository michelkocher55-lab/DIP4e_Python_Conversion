from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from General.edge import edge
from libDIPUM.splitmerge import splitmerge
from libDIPUM.data_path import dip_data

# Data
path_f = dip_data("constant-gray-region.tif")
path_fn = dip_data("Fig1001(d)(noisy_region).tif")

f = imread(path_f)
fn = imread(path_fn)

# Edge detection on original
# gedge = edge (f, 'sobel', 0);
# In MATLAB, threshold 0 means usually default or strict 0.
# Our edge.py handles 0 explicitly if passed, but let's see.
# If 0 implies "all gradients > 0", it will be noisy if image isn't perfect constant.
# But f is "constant-gray-region", so it has perfect regions.
# Sobel will find the boundary.
gedge = edge(f, "sobel", threshold=None)  # Start with auto

# Thresholding
# mx = max(f(:)); mn = min(f(:)); T = mn + (mx - mn)/2; gthresh = f > T;
mx = f.max()
mn = f.min()
T = mn + (mx - mn) / 2
gthresh = f > T

# Noisy region
# gnedge = edge(fn, 'sobel', 0);
gnedge = edge(fn, "sobel", threshold=None)


# Splitmerge
# gsm = splitmerge(fn, 8, @predicate2);
# predicate2: flag = (sd > 10);
def predicate2(region: Any):
    """predicate2."""
    if region.size == 0:
        return False
    sd = np.std(region)
    return sd > 10


gsm = splitmerge(fn, 8, predicate2)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("f")
axes[0].axis("off")

axes[1].imshow(gedge, cmap="gray")
axes[1].set_title("Edge (Sobel)")
axes[1].axis("off")

axes[2].imshow(gthresh, cmap="gray")
axes[2].set_title("Thresholded")
axes[2].axis("off")

axes[3].imshow(fn, cmap="gray")
axes[3].set_title("Noisy Image")
axes[3].axis("off")

axes[4].imshow(gnedge, cmap="gray")
axes[4].set_title("Edge of Noisy Image")
axes[4].axis("off")

axes[5].imshow(gsm, cmap="jet")  # Color to show segmentation
axes[5].set_title("SplitMerge Result")
axes[5].axis("off")

plt.tight_layout()
plt.savefig("Figure101.png")
print("Saved Figure101.png")
plt.show()
