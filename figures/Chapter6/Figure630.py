import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from libDIPUM.wavefast import wavefast
from libDIPUM.wavedisplay import wavedisplay
from libDIPUM.data_path import dip_data

# Data
f = img_as_float(imread(dip_data("Vase.tif")))

# Fast wavelet transform
c, s = wavefast(f, 1, "haar")
c2, s2 = wavefast(f, 2, "haar")
c8, s8 = wavefast(f, 8, "haar")

# Display
plt.figure()
plt.imshow(f, cmap="gray")
plt.axis("off")
plt.savefig("Figure630.png")

plt.figure()
w = wavedisplay(c, s)
plt.imshow(w, cmap="gray")
plt.axis("off")
plt.savefig("Figure630Bis.png")

plt.figure()
w2 = wavedisplay(c2, s2)
plt.imshow(w2, cmap="gray")
plt.axis("off")
plt.savefig("Figure630Ter.png")

plt.figure()
Temp = wavedisplay(c8, s8, 4)
plt.imshow(Temp, cmap="gray")
plt.axis("off")
plt.savefig("Figure630Quart.png")

plt.show()
