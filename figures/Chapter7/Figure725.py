import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# %% Data
base = "/Users/michelkocher/michel/Data/DIP-DIPUM/DIP"
fIR = imread(f"{base}/WashingtonDC-Band4-NearInfrared-512.tif")
fR = imread(f"{base}/WashingtonDC-Band3-Red-512.tif")
fG = imread(f"{base}/WashingtonDC-Band2-Green-512.tif")
fB = imread(f"{base}/WashingtonDC-Band1-Blue-512.tif")

# %% Substitute Red by Infrared
fRtoIR = np.stack((fIR, fG, fB), axis=2)

# %% Substitute Green by Infrared
fGtoIR = np.stack((fR, fIR, fB), axis=2)

# %% Display
plt.figure(figsize=(10, 7))

plt.subplot(2, 3, 1)
plt.imshow(fR, cmap="gray")
plt.axis("off")
plt.title("Red channel")

plt.subplot(2, 3, 2)
plt.imshow(fG, cmap="gray")
plt.axis("off")
plt.title("Green channel")

plt.subplot(2, 3, 3)
plt.imshow(fB, cmap="gray")
plt.axis("off")
plt.title("Blue channel")

plt.subplot(2, 3, 4)
plt.imshow(fIR, cmap="gray")
plt.axis("off")
plt.title("Infra red channel")

plt.subplot(2, 3, 5)
plt.imshow(fRtoIR)
plt.axis("off")
plt.title("Infra Red -> Red")

plt.subplot(2, 3, 6)
plt.imshow(fGtoIR)
plt.axis("off")
plt.title("Infra red -> Green")

plt.tight_layout()
plt.savefig("Figure725.png")
plt.show()
