import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.wavefast import wavefast
from libDIPUM.wavedisplay import wavedisplay
from libDIPUM.data_path import dip_data

# Parameters
n_levels = 3

# Data
f = imread(dip_data("lena.tif"))
if f.ndim == 3:
    f = f[..., 0]

# Fast wavelet transform
c1, s1 = wavefast(f, n_levels, "haar")
c2, s2 = wavefast(f, n_levels, "db4")
c3, s3 = wavefast(f, n_levels, "sym4")
c4, s4 = wavefast(f, n_levels, "bior6.8")

# Display + save
plt.figure(1)
plt.imshow(wavedisplay(c1, s1), cmap="gray")
plt.axis("off")
plt.savefig("Figure843.png", dpi=150, bbox_inches="tight")

plt.figure(2)
plt.imshow(wavedisplay(c2, s2), cmap="gray")
plt.axis("off")
plt.savefig("Figure843Bis.png", dpi=150, bbox_inches="tight")

plt.figure(3)
plt.imshow(wavedisplay(c3, s3), cmap="gray")
plt.axis("off")
plt.savefig("Figure843Ter.png", dpi=150, bbox_inches="tight")

plt.figure(4)
plt.imshow(wavedisplay(c4, s4), cmap="gray")
plt.axis("off")
plt.savefig("Figure843Quart.png", dpi=150, bbox_inches="tight")

plt.show()
