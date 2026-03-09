import matplotlib.pyplot as plt
from skimage.io import imread
from helpers.ReconstructionUsingBitPlanes import ReconstructionUsingBitPlanes
from libDIPUM.data_path import dip_data

# Image loading
img_name = dip_data("trophozoite.tif")
f = imread(img_name)
if f.ndim == 3:
    f = f[:, :, 0]

NBits = 8

# 1. Bit Plane Slicing
# "BitPlanes" needs to be prepared such that index 0 is MSB (iter=1 in MATLAB)
# MATLAB loop i=1:8. Bit(:,:,i) = ... 2^(NBits-i).
# i=1 -> 2^7.

BitPlanes = []
for i in range(NBits):  # 0..7
    # Power of 2:
    power = NBits - 1 - i
    # i=0 (iter 1) -> 7. 2^7.

    plane = (f >> power) & 1
    BitPlanes.append(plane)

# 2. Reconstruction
Recs = []
SNRs = []

for iter_count in range(1, NBits + 1):
    # Use first 'iter_count' planes
    Rec, val = ReconstructionUsingBitPlanes(f, BitPlanes, NBits, iter_count)
    Recs.append(Rec)
    SNRs.append(val)

# Display
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

for i in range(NBits):
    ax = axes[i]
    rec_img = Recs[i]
    snr_v = SNRs[i]

    ax.imshow(rec_img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{i + 1} MSB planes\nSNR = {snr_v:.2f} [dB]")
    ax.axis("off")

plt.tight_layout()
plt.savefig("Figure315.png")
print("Saved Figure315.png")
plt.show()
