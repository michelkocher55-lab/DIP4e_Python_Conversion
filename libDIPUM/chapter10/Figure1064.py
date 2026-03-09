import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import ia870 as ia
from libDIPUM.data_path import dip_data

# Figure1064

# Data
a = imread(dip_data("corneacells.tif"))

# Image cleaning
c = ia.iaasf(a, "oc", ia.iasecross(), 2)

# Internal marker
d = ia.iaregmax(c)

# External marker obtained by watershed
e = ia.ianeg(a)  # Segmentation function
f = ia.iacwatershed(e, d)

# Internal + External Marker
# As the internal and external markers can be touching, we separate them by using the concept of labelled image

h1 = ia.iaaddm(ia.ialabel(d), np.uint16(1))  # BG = 1, First particle = 2
h2 = ia.iagray(d, "uint16")  # BG = 0, prticle = 65535
h = ia.iaintersec(h2, h1)  # BG = 0, First particle = 2

g = ia.iagray(f, "uint16", 1)  # External Marker = 1
i = ia.iaunion(g, h)  # BG = 0, External Marker = 1, First particle = 2

# Segmentation function
j = ia.iagradm(a)

# Constrained watershed of the gradient from markers
# Apply the constrained watershed on the gradient from the labeled internal and external markers.
k = ia.iacwatershed(j, i)

# Display

(fig, axes) = plt.subplots(nrows=2, ncols=2)
axes[0, 0].set_title("a")
axes[0, 0].imshow(a, cmap="gray")
axes[0, 0].axis("off")
axes[0, 1].set_title("c")
axes[0, 1].imshow(c, cmap="gray")
axes[0, 1].axis("off")
axes[1, 0].set_title("d")
axes[1, 0].imshow(MKRLib.mmshow(a, d), cmap="gray")
axes[1, 0].axis("off")
axes[1, 1].set_title("e,f,d")
axes[1, 1].imshow((MKRLib.mmshow(e, f, d)), cmap="gray")
axes[1, 1].axis("off")
fig.tight_layout()
plt.savefig("Tiadcornea.png")

(fig, axes) = plt.subplots(nrows=2, ncols=2)
axes[0, 0].set_title("h1")
# axes[0, 0].imshow(h1, cmap=CMap)
axes[0, 0].imshow(MKRLib.mmlblshow(h1))
axes[0, 0].axis("off")
axes[0, 1].set_title("h2")
axes[0, 1].imshow(h2, cmap="gray")
axes[0, 1].axis("off")
axes[1, 0].set_title("h")
axes[1, 0].imshow(MKRLib.mmlblshow(h))
axes[1, 0].axis("off")
axes[1, 1].set_title("g")
axes[1, 1].imshow(g, cmap="gray")
axes[1, 1].axis("off")
fig.tight_layout()
plt.savefig("TiadcorneaBis.png")

(fig, axes) = plt.subplots(nrows=1, ncols=1)
axes.set_title("i")
axes.imshow(MKRLib.mmlblshow(i))
axes.axis("off")
fig.tight_layout()
plt.savefig("TiadcorneaTer.png")

(fig, axes) = plt.subplots(nrows=1, ncols=1)
axes.set_title("a,d,k")
axes.imshow(MKRLib.mmshow(a, d > 0, k > 0), cmap="gray")
axes.axis("off")
fig.tight_layout()
plt.savefig("TiadcorneaQuart.png")
