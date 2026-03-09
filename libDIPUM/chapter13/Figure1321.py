from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from libDIPUM.imstack2vectors import imstack2vectors
from libDIPUM.covmatrix import covmatrix
from libDIPUM.bayesgauss import bayesgauss
from libDIPUM.data_path import dip_data

print("Running Figure1321 (Bayes classification of remote data)...")


def _imread_gray(path: str) -> np.ndarray:
    """_imread_gray."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


# Read masks
B1 = _imread_gray(dip_data("WashingtonDC-mask-water-512.tif"))
B2 = _imread_gray(dip_data("WashingtonDC-mask-urban-512.tif"))
B3 = _imread_gray(dip_data("WashingtonDC-mask-vegetation-512.tif"))

# Read multispectral images
f1 = _imread_gray(dip_data("WashingtonDC-Band1-Blue-512.tif"))
f2 = _imread_gray(dip_data("WashingtonDC-Band2-Green-512.tif"))
f3 = _imread_gray(dip_data("WashingtonDC-Band3-Red-512.tif"))
f4 = _imread_gray(dip_data("WashingtonDC-Band4-NearInfrared-512.tif"))

# Stack images
stack = np.dstack((f1, f2, f3, f4))

# Extract pattern vectors in mask regions
X1, R1 = imstack2vectors(stack, B1)
X2, R2 = imstack2vectors(stack, B2)
X3, R3 = imstack2vectors(stack, B3)

# Training patterns (odd rows in MATLAB -> Python step 2 from index 0)
T1 = X1[0::2, :]
T2 = X2[0::2, :]
T3 = X3[0::2, :]

# Mean vectors and covariance matrices
C1, m1 = covmatrix(T1)
C2, m2 = covmatrix(T2)
C3, m3 = covmatrix(T3)

# Arrays for bayesgauss
CA = np.dstack((C1, C2, C3))
MA = np.hstack((m1, m2, m3)).T

# Classify training set
dT1 = bayesgauss(T1, CA, MA)
dT2 = bayesgauss(T2, CA, MA)
dT3 = bayesgauss(T3, CA, MA)

# Training counts
Class1_to_1_T = int(np.sum(dT1 == 1))
Class1_to_2_T = int(np.sum(dT1 == 2))
Class1_to_3_T = int(np.sum(dT1 == 3))
Class2_to_1_T = int(np.sum(dT2 == 1))
Class2_to_2_T = int(np.sum(dT2 == 2))
Class2_to_3_T = int(np.sum(dT2 == 3))
Class3_to_1_T = int(np.sum(dT3 == 1))
Class3_to_2_T = int(np.sum(dT3 == 2))
Class3_to_3_T = int(np.sum(dT3 == 3))

# Independent data (even rows in MATLAB -> Python step 2 from index 1)
I1 = X1[1::2, :]
I2 = X2[1::2, :]
I3 = X3[1::2, :]

# Classify testing set
dI1 = bayesgauss(I1, CA, MA)
dI2 = bayesgauss(I2, CA, MA)
dI3 = bayesgauss(I3, CA, MA)

# Testing counts
Class1_to_1 = int(np.sum(dI1 == 1))
Class1_to_2 = int(np.sum(dI1 == 2))
Class1_to_3 = int(np.sum(dI1 == 3))
Class2_to_1 = int(np.sum(dI2 == 1))
Class2_to_2 = int(np.sum(dI2 == 2))
Class2_to_3 = int(np.sum(dI2 == 3))
Class3_to_1 = int(np.sum(dI3 == 1))
Class3_to_2 = int(np.sum(dI3 == 2))
Class3_to_3 = int(np.sum(dI3 == 3))

print("Training confusion-like counts:")
print(f"Class1 -> [1,2,3] = [{Class1_to_1_T}, {Class1_to_2_T}, {Class1_to_3_T}]")
print(f"Class2 -> [1,2,3] = [{Class2_to_1_T}, {Class2_to_2_T}, {Class2_to_3_T}]")
print(f"Class3 -> [1,2,3] = [{Class3_to_1_T}, {Class3_to_2_T}, {Class3_to_3_T}]")
print("Testing confusion-like counts:")
print(f"Class1 -> [1,2,3] = [{Class1_to_1}, {Class1_to_2}, {Class1_to_3}]")
print(f"Class2 -> [1,2,3] = [{Class2_to_1}, {Class2_to_2}, {Class2_to_3}]")
print(f"Class3 -> [1,2,3] = [{Class3_to_1}, {Class3_to_2}, {Class3_to_3}]")

# Classify all pixels in whole image region
B = np.ones_like(f1)
X, R = imstack2vectors(stack, B)
dAll = bayesgauss(X, CA, MA)

# Rebuild class images using MATLAB/Fortran linear indexing order.
M, N = f1.shape
class1_vec = np.zeros(M * N, dtype=np.uint8)
class2_vec = np.zeros(M * N, dtype=np.uint8)
class3_vec = np.zeros(M * N, dtype=np.uint8)

class1_vec[np.where(dAll == 1)[0]] = 1
class2_vec[np.where(dAll == 2)[0]] = 1
class3_vec[np.where(dAll == 3)[0]] = 1

class1 = np.reshape(class1_vec, (M, N), order="F")
class2 = np.reshape(class2_vec, (M, N), order="F")
class3 = np.reshape(class3_vec, (M, N), order="F")

# Display
fig = plt.figure(1)

plt.subplot(3, 3, 1)
plt.imshow(f1, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(f2, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(f3, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(f4, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow((B1 != 0) | (B2 != 0) | (B3 != 0), cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(class1, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(class2, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(class3, cmap="gray")
plt.axis("off")

out_path = os.path.join(os.path.dirname(__file__), "Figure1321.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
