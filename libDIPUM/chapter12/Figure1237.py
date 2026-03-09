import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import rotate

from libDIPUM.invmoments import invmoments
from libDIPUM.data_path import dip_data

print("Running Figure1237 (Invariant moments under transformations)...")

# Parameters
Offset = 150
r = 2
Theta1 = 45
Theta2 = 90

# Data
image_path = dip_data("Fig1123(a)(Original_Padded_to_568_by_568).tif")
fP = imread(image_path)
if fP.ndim == 3:
    fP = fP[:, :, 0]

NR, NC = fP.shape
Motif = fP[84:484, 84:484]  # MATLAB 85:484

fT = np.zeros_like(fP, dtype=np.uint8)
h, w = Motif.shape
fT[Offset : Offset + h, Offset : Offset + w] = Motif

fHS = Motif[::r, ::r]
fHSP = np.pad(fHS, ((184, 184), (184, 184)), mode="constant", constant_values=0)

fM = np.fliplr(Motif)
fMP = np.pad(fM, ((84, 84), (84, 84)), mode="constant", constant_values=0)

fR45 = rotate(
    Motif.astype(float),
    Theta1,
    reshape=True,
    order=1,
    mode="constant",
    cval=0.0,
    prefilter=False,
)
fR45 = np.clip(np.rint(fR45), 0, 255).astype(np.uint8)

fR90 = rotate(
    Motif.astype(float),
    Theta2,
    reshape=True,
    order=1,
    mode="constant",
    cval=0.0,
    prefilter=False,
)
fR90 = np.clip(np.rint(fR90), 0, 255).astype(np.uint8)
fR90P = np.pad(fR90, ((84, 84), (84, 84)), mode="constant", constant_values=0)

# Invariant moments
imgs = [fP, fT, fHSP, fMP, fR45, fR90P]
Phi = np.zeros((6, 7), dtype=float)
for i, img in enumerate(imgs):
    m = invmoments(img)
    # MATLAB expression: abs(log10(invmoments(...))).
    # Use complex log to preserve sign information for negative moments.
    mc = m.astype(np.complex128)
    mc[m == 0] = np.finfo(float).eps
    Phi[i, :] = np.abs(np.log10(mc))

# Display phi transpose in console
print("Phi.T =")
print(Phi.T)

# Figure 1
fig1, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.ravel()

ax[0].imshow(fP, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(fT, cmap="gray")
ax[1].set_title(f"Translation, dk = dl = {Offset}")
ax[1].axis("off")

ax[2].imshow(fHSP, cmap="gray")
ax[2].set_title(f"Resizing, r = {r}")
ax[2].axis("off")

ax[3].imshow(fMP, cmap="gray")
ax[3].set_title("Mirroring")
ax[3].axis("off")

ax[4].imshow(fR45, cmap="gray")
ax[4].set_title(f"Rotation, theta = {Theta1}")
ax[4].axis("off")

ax[5].imshow(fR90P, cmap="gray")
ax[5].set_title(f"Rotation, theta = {Theta2}")
ax[5].axis("off")

fig1.tight_layout()
fig1.savefig("Figure1237.png")

# Figure 2
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
ax2.plot(Phi, "o-")
ax2.set_xlabel("Transformation")
ax2.set_ylabel("Phi_i")
ax2.set_title("Moments")
ax2.set_xticks(np.arange(6))
ax2.set_xticklabels(
    ["None", "Translation", "Scaling", "Mirror", "Rotation45", "Rotation90"]
)
ax2.legend(["Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5", "Phi_6", "Phi_7"])

fig2.tight_layout()
fig2.savefig("Figure1237Bis.png")

print("Saved Figure1237.png and Figure1237Bis.png")
plt.show()
