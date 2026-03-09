from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.im2minperpoly import im2minperpoly
from libDIPUM.randvertex import randvertex
from libDIPUM.polyangles import polyangles
from libDIPUM.strsimilarity import strsimilarity
from libDIPUM.data_path import dip_data

# Parameters
Nr = 10
q = 45
MaxDeviation = 9

# Data
path_f1 = dip_data("Fig1203(a)(bottle_1).tif")
path_f2 = dip_data("Fig1203(d)(bottle_2).tif")

f1 = imread(path_f1)
f2 = imread(path_f2)

# MPP
print("Computing MPP for f1...")
X1, Y1, R1 = im2minperpoly(f1, 8)
print(f"MPP1 vertices: {len(X1)}")

print("Computing MPP for f2...")
X2, Y2, R2 = im2minperpoly(f2, 8)
print(f"MPP2 vertices: {len(X2)}")

# Noise Adding (Skipping full Nr loop for visual check first, or just do one)
Xn1_list, Yn1_list = [], []
Xn2_list, Yn2_list = [], []

np.random.seed(0)  # rng default

for r in range(Nr):
    xn1, yn1 = randvertex(X1, Y1, MaxDeviation)
    Xn1_list.append(xn1)
    Yn1_list.append(yn1)

    xn2, yn2 = randvertex(X2, Y2, MaxDeviation)
    Xn2_list.append(xn2)
    Yn2_list.append(yn2)

# Signature Computation
Angles1 = polyangles(X1, Y1)
Angles2 = polyangles(X2, Y2)

# Store noisy angles
AnglesN1 = []
AnglesN2 = []

for r in range(Nr):
    temp1 = polyangles(Xn1_list[r], Yn1_list[r])
    AnglesN1.append(temp1)

    temp2 = polyangles(Xn2_list[r], Yn2_list[r])
    AnglesN2.append(temp2)

# String conversion


def vec2str(v: Any):
    """vec2str."""
    # Emulate int2str behavior for vector
    # "1  2  3"
    return "  ".join([str(int(x)) for x in v])


s1_vec = np.floor(Angles1 / q) + 1
s1 = vec2str(s1_vec)

s2_vec = np.floor(Angles2 / q) + 1
s2 = vec2str(s2_vec)

sN1 = [vec2str(np.floor(a / q) + 1) for a in AnglesN1]
sN2 = [vec2str(np.floor(a / q) + 1) for a in AnglesN2]

# Similarity
R12, _, _ = strsimilarity(s1, s2)
print(f"Similarity s1-s2: {R12}")

R1N1 = []
for r in range(Nr):
    val, _, _ = strsimilarity(s1, sN1[r])
    R1N1.append(val)

print(f"Mean Similarity s1-sN1: {np.mean(R1N1)}")

# Display
plt.figure(figsize=(15, 10))

# Subplot 1: f1
plt.subplot(2, 3, 1)
plt.imshow(f1, cmap="gray")
plt.title("f1")

# Subplot 2: Bound f1
plt.subplot(2, 3, 2)
plt.plot(np.append(Y1, Y1[0]), np.append(X1, X1[0]), ".-")
plt.gca().invert_yaxis()  # Match image coords
plt.title(f"Bound f1, {len(X1)}")
plt.axis("equal")

# Subplot 3: Noisy f1
plt.subplot(2, 3, 3)
xn_disp = Xn1_list[0]
yn_disp = Yn1_list[0]
plt.plot(np.append(yn_disp, yn_disp[0]), np.append(xn_disp, xn_disp[0]), ".-")
plt.gca().invert_yaxis()
plt.title(f"Noisy f1, {len(xn_disp)}")
plt.axis("equal")

# Subplot 4: f2
plt.subplot(2, 3, 4)
plt.imshow(f2, cmap="gray")
plt.title("f2")

# Subplot 5: Bound f2
plt.subplot(2, 3, 5)
plt.plot(np.append(Y2, Y2[0]), np.append(X2, X2[0]), ".-")
plt.gca().invert_yaxis()
plt.title(f"Bound f2, {len(X2)}")
plt.axis("equal")

# Subplot 6: Noisy f2
plt.subplot(2, 3, 6)
xn_disp2 = Xn2_list[0]
yn_disp2 = Yn2_list[0]
plt.plot(np.append(yn_disp2, yn_disp2[0]), np.append(xn_disp2, xn_disp2[0]), ".-")
plt.gca().invert_yaxis()
plt.title(f"Noisy f2, {len(xn_disp2)}")
plt.axis("equal")

plt.tight_layout()
plt.savefig("Figure1308.png")
print("Figure saved to Figure1308.png")
plt.show()
