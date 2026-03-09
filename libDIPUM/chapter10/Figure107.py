import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.ndimage import convolve
from libDIP.intScaling4e import intScaling4e
from libDIPUM.pixeldup import pixeldup
from libDIPUM.data_path import dip_data

# Data
f = img_as_float(imread(dip_data("Fig1007(a)(wirebond_mask).tif")))
M, N = f.shape

# Kernel
w = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)

# Filtering
g = convolve(f, w, mode="nearest")

# Top Left Corner Zoom
h_top = int(M / 4)
w_top = int(N / 4)

gtop = g[0:h_top, 0:w_top]
# gtop = pixeldup(gtop, 4);
gtop = pixeldup(gtop, 4)

# Crop back to size of f
gtop = gtop[:M, :N]

# Bottom Right Corner Zoom
# gbot = g(end - int32(M/4) + 1:end, end - int32(N/4) + 1:end);
# Python: g[M-h_top : M, N-w_top : N]
gbot = g[M - h_top : M, N - w_top : N]
gbot = pixeldup(gbot, 4)
gbot = gbot[:M, :N]

botmax = gbot.max()
gtop[-1, -1] = botmax

# Positive values
gpos = np.zeros_like(g)
gpos[g > 0] = g[g > 0]

# Thresholding
T = gpos.max()
gt = gpos >= T

# Scaling for display
gs = intScaling4e(g)
gtop_s = intScaling4e(gtop)
gbot_s = intScaling4e(gbot)
gpos_s = intScaling4e(gpos)
gt_s = intScaling4e(gt)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(f, cmap="gray")
axes[0].set_title("f")
axes[0].axis("off")

axes[1].imshow(gs, cmap="gray")
axes[1].set_title("g (Filtered)")
axes[1].axis("off")

axes[2].imshow(gtop_s, cmap="gray")
axes[2].set_title("Top Corner Zoom (gtop)")
axes[2].axis("off")

axes[3].imshow(gbot_s, cmap="gray")
axes[3].set_title("Bottom Corner Zoom (gbot)")
axes[3].axis("off")

axes[4].imshow(gpos_s, cmap="gray")
axes[4].set_title("Positive Values (gpos)")
axes[4].axis("off")

axes[5].imshow(gt_s, cmap="gray")
axes[5].set_title("Thresholded (gt)")
axes[5].axis("off")

plt.tight_layout()
plt.savefig("Figure107.png")
plt.show()

if __name__ == "__main__":
    Figure107()
