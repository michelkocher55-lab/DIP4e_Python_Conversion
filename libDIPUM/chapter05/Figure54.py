import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.imnoise2New import imnoise2New
from PIL import Image
from libDIPUM.data_path import dip_data

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
a = [0.15, 0, 5, 10, 0, 0.05]
b = [0.07, 0.03, 1, np.nan, 0.3, 0.05]

NBin = 256
Bin = np.linspace(0, 1, NBin)
BinCenters = 0.5 * (Bin[:-1] + Bin[1:])
bin_width = Bin[1] - Bin[0]
# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
f = np.array(Image.open(dip_data("test-pattern.tif")), dtype=np.float64)

# im2double
if f.max() > 1.0:
    f /= 255.0

# f(1:2:end,1:2:end)  → Python slicing
f = f[::2, ::2]
M, N = f.shape


# ------------------------------------------------------------
# Add noise
# ------------------------------------------------------------
fn1, r = imnoise2New(f, "gaussian", a[0], b[0])
fn2, r = imnoise2New(f, "rayleigh", a[1], b[1])
fn3, r = imnoise2New(f, "erlang", a[2], b[2])
fn4, r = imnoise2New(f, "exponential", a[3], b[3])
fn5, r = imnoise2New(f, "uniform", a[4], b[4])
fn6, r = imnoise2New(f, "salt & pepper", a[5], b[5])


# ------------------------------------------------------------
# Figure 1
# ------------------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(fn1, cmap="gray")
plt.title(f"Gauss, μ = {a[0]}, σ = {b[0]}")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(fn2, cmap="gray")
plt.title(f"Rayleigh, μ = {a[1]}, σ = {b[1]}")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(fn3, cmap="gray")
plt.title(f"Erlang, μ = {a[2]}, σ = {b[2]}")
plt.axis("off")

plt.subplot(2, 3, 4)
Hist, edges = np.histogram(fn1.ravel(), bins=256, range=(0, 1))
Hist = Hist[1:-1]
edges = edges[1:-1]
plt.plot(edges[:-1], Hist)

plt.subplot(2, 3, 5)
Hist, edges = np.histogram(fn2.ravel(), bins=256, range=(0, 1))
Hist = Hist[1:-1]
edges = edges[1:-1]
plt.plot(edges[:-1], Hist)

plt.subplot(2, 3, 6)
Hist, edges = np.histogram(fn3.ravel(), bins=256, range=(0, 1))
Hist = Hist[1:-1]
edges = edges[1:-1]
plt.plot(edges[:-1], Hist)

plt.tight_layout()
plt.savefig("Figure54.png", dpi=300, bbox_inches="tight")
plt.show()
# plt.close()


# ------------------------------------------------------------
# Figure 2
# ------------------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(fn4, cmap="gray")
plt.title(f"Exponential, a = {a[3]}")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(fn5, cmap="gray")
plt.title(f"Uniform, a = {a[4]}, b = {b[4]}")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(fn6, cmap="gray")
plt.title(f"Salt & Pepper, a = {a[5]}, b = {b[5]}")
plt.axis("off")

plt.subplot(2, 3, 4)
Hist, edges = np.histogram(fn4.ravel(), bins=256, range=(0, 1))
Hist = Hist[1:-1]
edges = edges[1:-1]
plt.plot(edges[:-1], Hist)

plt.subplot(2, 3, 5)
Hist, edges = np.histogram(fn5.ravel(), bins=256, range=(0, 1))
Hist = Hist[1:-1]
edges = edges[1:-1]
plt.plot(edges[:-1], Hist)

plt.subplot(2, 3, 6)
Hist, edges = np.histogram(fn6.ravel(), bins=256, range=(0, 1))
Hist = Hist[1:-1]
edges = edges[1:-1]
plt.plot(edges[:-1], Hist)

plt.tight_layout()
plt.savefig("Figure54Bis.png", dpi=300, bbox_inches="tight")
plt.show()
# plt.close()
