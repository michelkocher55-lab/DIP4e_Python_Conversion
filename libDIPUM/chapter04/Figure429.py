import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIP.lpFilterTF4e import lpFilterTF4e
from libDIP.dftFiltering4e import dftFiltering4e
from libDIPUM.data_path import dip_data

# Image loading
img_path = dip_data("blown_ic.tif")
f = imread(img_path)
if f.ndim == 3:
    f = f[:, :, 0]  # if grayscale
f = img_as_float(f)
M, N = f.shape

# Low pass filter creation (Centered)
# HNoPad = lpFilterTF4e('ideal',M,N,0.5);
# HNoPad = imcomplement(HNoPad); -> 1 - H
HNoPad_low = lpFilterTF4e("ideal", M, N, 0.5)
HNoPad = 1.0 - HNoPad_low

# With Padding
# PQ = paddedsize(size(f)); -> dftFiltering4e pads to 2*M, 2*N internally if padmode != 'none'
# Wait, dftFiltering4e documentation says:
# "Unless padmode = 'none', function DFTFILTERING4E pads the input image to size P-by-Q, with P = 2*M and Q = 2*N"
# So we must generate H of size 2M, 2N.

P, Q = 2 * M, 2 * N
HPad_low = lpFilterTF4e("ideal", P, Q, 0.5)
HPad = 1.0 - HPad_low

# Filtering

# NO PADDING
# gNoPad=dftfilt(f,HNoPad); -> dftFiltering4e(f, HNoPad, 'none')
gNoPad = dftFiltering4e(f, HNoPad, "none")

# WITH PADDING
# gPad=dftfilt(f,HPad); -> dftFiltering4e(f, HPad, 'zeros' or 'replicate'?)
# Input script Figure429.m uses `lpFilterTF4e('ideal',PQ(1),PQ(2),0.5)` where PQ=paddedsize(size(f)).
# And calls gPad=dftfilt(f,HPad). dftfilt uses 'zeros' (constant) padding by default.
# dftFiltering4e defaults to 'replicate'.
# We should use 'zeros' to match Figure429.m logic?
# "gPad=dftfilt(f,HPad)" in Figure429.m calls dftfilt.m.
# dftfilt.m default padMethod is 'zeros' (0).
# So I should use 'zeros' in dftFiltering4e to replicate behavior.

gPad = dftFiltering4e(f, HPad, "zeros")

# Stats
print(
    f"No Pad: Min={gNoPad.min():.4f}, Max={gNoPad.max():.4f}, Mean={gNoPad.mean():.4e}"
)
print(f"With Pad: Min={gPad.min():.4f}, Max={gPad.max():.4f}, Mean={gPad.mean():.4f}")

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(f, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(gNoPad, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("HighPass No Padding")
axes[1].axis("off")

axes[2].imshow(gPad, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("HighPass With Padding")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("Figure429.png")
print("Saved Figure429.png")
plt.show()
