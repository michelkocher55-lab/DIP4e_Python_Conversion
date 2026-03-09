from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.lpFilterTF4e import lpFilterTF4e
from libDIP.hpFilterTF4e import hpFilterTF4e
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data


def mesh_plot(ax: Any, H: Any):
    """mesh_plot."""
    # Helper to plot mesh similar to MATLAB
    # Subsample to match MATLAB 1:10:500
    step = 10
    H_sub = H[:500:step, :500:step]
    x = np.arange(0, 500, step)
    y = np.arange(0, 500, step)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Wireframe only (no filled faces, no shading)
    ax.plot_wireframe(X, Y, H_sub, color="black", linewidth=0.4)
    ax.view_init(elev=45, azim=45)  # Adjust view
    ax.axis("off")


print("Running Figure430 (Highpass Filtering)...")

# Image loading
img_name = dip_data("blown_ic_crop.tif")
f = imread(img_name)
if f.ndim == 3:
    f = f[:, :, 0]

f = img_as_float(f)
M, N = f.shape

# Mesh Lowpass for display (500x500, D0=50)
# HLPDisp = lpFilterTF4e('gaussian', 500, 500, 50)
HLPDisp = lpFilterTF4e("gaussian", 500, 500, 50)

# Mesh Highpass
HHPDisp = hpFilterTF4e("gaussian", 500, 500, 50)

# Filtering
# Lowpass: D0=10, 2M x 2N
# HLP = lpFilterTF4e('gaussian', 2*M, 2*N, 10)
HLP = lpFilterTF4e("gaussian", 2 * M, 2 * N, 10)
glow = dftFiltering4e(f, HLP)

# Highpass: D0=10
# HHP = hpFilterTF4e('gaussian', 2*M, 2*N, 10)
HHP = hpFilterTF4e("gaussian", 2 * M, 2 * N, 10)
ghigh = dftFiltering4e(f, HHP)

# High Frequency Emphasis
# Hemphasis = 0.85 + hpFilterTF4e('gaussian', 2*M, 2*N, 10)
Hemphasis = 0.85 + hpFilterTF4e("gaussian", 2 * M, 2 * N, 10)
gemph = dftFiltering4e(f, Hemphasis)

# Display
fig = plt.figure(figsize=(15, 10))

# 1. HLP Mesh
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
mesh_plot(ax1, HLPDisp)
ax1.set_title("Gaussian LP Transfer Function")

# 2. HHP Mesh
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
mesh_plot(ax2, HHPDisp)
ax2.set_title("Gaussian HP Transfer Function")

# 3. Duplicate HHP Mesh? Figure430.m has duplicate? "subplot(2,3,3) mesh(HHPDisp)"
# Ah, maybe it was meant for Emphasis filter or just repeated?
# Original code: subplot(2,3,3) mesh(HHPDisp).
# I will replicate.
ax3 = fig.add_subplot(2, 3, 3, projection="3d")
mesh_plot(ax3, HHPDisp)
ax3.set_title("Gaussian HP (Repeated)")

# 4. Lowpass result
ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(glow, cmap="gray", vmin=0, vmax=1)
ax4.set_title("Lowpass Filtered")
ax4.axis("off")

# 5. Highpass result
ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(ghigh, cmap="gray", vmin=0, vmax=1)
ax5.set_title("Highpass Filtered")
ax5.axis("off")

# 6. Emphasis result
ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(gemph, cmap="gray", vmin=0, vmax=1)
ax6.set_title("High Freq Emphasis")
ax6.axis("off")

plt.tight_layout()
plt.savefig("Figure430.png")
print("Saved Figure430.png")
plt.show()
