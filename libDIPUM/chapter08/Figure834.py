from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.motion import motion
from libDIPUM.data_path import dip_data


def _invert_if_needed(img: Any):
    """_invert_if_needed."""
    # Match MATLAB visual polarity for these NASA TIFF frames.
    if np.issubdtype(img.dtype, np.integer):
        return np.iinfo(img.dtype).max - img
    return 1.0 - img


# %% Figure 834

# %% Data
j = imread(dip_data("nasa67.tif"))
if j.ndim == 3:
    j = j[:, :, 0]
# MATLAB: j = j(1:352, 119:470)
j = j[0:352, 118:470]
j = _invert_if_needed(j)

i = imread(dip_data("nasa79.tif"))
if i.ndim == 3:
    i = i[:, :, 0]
# MATLAB: i = i(1:352, 119:470)
i = i[0:352, 118:470]
i = _invert_if_needed(i)

# %% Motion computation
e, a, dx, dy = motion(i, j, 16, [16, 16], 1)

# %% Display
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(j, cmap="gray")
plt.title("Frame 67")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(i, cmap="gray")
plt.title("Frame 79")
plt.axis("off")

# mat2gray(double(i) - double(j))
d = i.astype(float) - j.astype(float)
dmin, dmax = np.min(d), np.max(d)
if dmax > dmin:
    dshow = (d - dmin) / (dmax - dmin)
else:
    dshow = np.zeros_like(d)

plt.subplot(2, 3, 4)
plt.imshow(dshow, cmap="gray", vmin=0, vmax=1)
plt.title("Difference")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(e, cmap="gray")  # MATLAB imshow(e, []) auto scaling
plt.title("column error")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(a, cmap="gray", vmin=0, vmax=255)
plt.title("Motion vectors")
plt.axis("off")

plt.tight_layout()
plt.savefig("Figure834.png")
print("Saved Figure834.png")
plt.show()
