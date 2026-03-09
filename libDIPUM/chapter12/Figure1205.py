import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import ia870
from scipy.ndimage import uniform_filter

from libDIPUM.bwboundaries import bwboundaries
from libDIPUM.freemanChainCode import freemanChainCode
from libDIPUM.bsubsamp import bsubsamp
from libDIPUM.connectpoly import connectpoly
from libDIPUM.data_path import dip_data

print("Running Figure1205...")

# 1. Data
path = dip_data("noisy-stroke.tif")
f = imread(path)
if f.ndim == 3:
    f = rgb2gray(f)

# 2. Denoising
f1 = uniform_filter(f.astype(float), size=9, mode="reflect")
f1 = np.clip(f1, 0, 255).astype(np.uint8)

# 3. Thresholding
thresh = threshold_otsu(f1)
bw = f1 > thresh

# Area opening
X1 = ia870.iaareaopen(bw, 100, ia870.iasebox())

# 5. Boundaries
# contours = find_contours(X1, 0.5)
contours = bwboundaries(X1, 8)

if len(contours) == 0:
    print("No contours found.")


# Longest boundary
longest_idx = np.argmax([len(c) for c in contours])
B = contours[longest_idx]

print(f"Longest boundary length: {len(B)}")

# 6. Boundary subsampling
r = 50
B1 = bsubsamp(B, 1 / r, f.shape[0], f.shape[1])

# Scale back
B1 = B1 * r

# 7. Connected polygon
B1C = connectpoly(B1[:, 0], B1[:, 1])

print(f"Connected polygon length: {len(B1C)}")

# 8. Freeman Chain Code
res = freemanChainCode(B1C, 8)
print("FCC computed. Length:", len(res.fcc))

# 9. Display
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
ax = axes.ravel()

# f
ax[0].imshow(f, cmap="gray")
ax[0].set_title(f"f, Size={f.shape}")

# f1
ax[1].imshow(f1, cmap="gray")
ax[1].set_title("f1 (Average 9x9)")

# X1
ax[2].imshow(X1, cmap="gray")
ax[2].set_title(f"X1 (Thresh={thresh:.1f}, Lambda=100)")

# N (original boundary, previously B)
ax[3].plot(B[:, 1], -B[:, 0], "r")
ax[3].axis("equal")
ax[3].set_title("N")

# B1 (subsampled boundary)
ax[4].plot(B1[:, 1], -B1[:, 0], "o", color="green", fillstyle="none")
ax[4].axis("equal")
ax[4].set_title(f"B1 (r={r})")

# B1C rendered as zero-order hold (horizontal/vertical only) between knots.
if len(B1) > 1:
    step_pts = [B1[0]]
    for p, q in zip(B1[:-1], B1[1:]):
        # Horizontal then vertical (Manhattan link)
        step_pts.append([p[0], q[1]])
        step_pts.append([q[0], q[1]])
    step_pts = np.asarray(step_pts)
else:
    step_pts = B1

ax[5].plot(step_pts[:, 1], -step_pts[:, 0], "b", linewidth=1.0)
ax[5].plot(B1[:, 1], -B1[:, 0], "o", color="black", markersize=3)
ax[5].axis("equal")
ax[5].set_title("B1C")

plt.tight_layout()
plt.savefig("Figure1205.png")
plt.show()
