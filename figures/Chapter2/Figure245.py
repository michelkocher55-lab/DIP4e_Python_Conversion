import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from scipy.fft import fft2, fftshift
from libDIPUM.imnoise3 import imnoise3
from libDIPUM.cnotch import cnotch
from libDIP.intScaling4e import intScaling4e
from libDIPUM.dftfilt import dftfilt
from libDIPUM.data_path import dip_data

# Data
img_path = dip_data("astronaut.tif")

# Load and convert to double
f = img_as_float(imread(img_path))
M, N = f.shape

# Generate sinusoidal noise
# [r, R, S] = imnoise3(M, N, [25 25], 0.3);
# Note: imnoise3 returns noise pattern r.
# C = [25, 25] means impulses at (u0, v0) = (25, 25) relative to center? or absolute?
# Imnoise3 logic usually takes C as coordinates of impulses.
# Let's assume the Python port follows MATLAB's imnoise3 signature: [r, R, S] = imnoise3(M, N, C, A, B)
# A=0.3 is amplitude.
r, R, S = imnoise3(M, N, [[25, 25]], [0.3])
# Note: imnoise3 implementation might vary. If it takes list of coords and list of Amps.
# Passing single list [25, 25] might define one impulse pair.

g = f + r

# Scale g for display/web as per MATLAB comment (improving contrast)
gs = intScaling4e(g)

# Compute spectrum
F = fft2(g)
G = fftshift(np.abs(F))
# Log transform for visualization
Glog = intScaling4e(1 + np.log(G))

# Create notch filters
# M=1650, N=2000 roughly? (Astronaut is likely smaller, 512x512?)
# Center is M/2, N/2 (0-indexed).
# MATLAB: M/2+1, N/2+1.
# MATLAB: Impulse at center + 25.
# Python: Center is M//2, N//2.
# Impulse location = (M//2 + 25, N//2 + 25).
# cnotch handles symmetry.

# H = cnotch('ideal', 'reject', M, N, [M/2+1+25, N/2+1+25], 2);
# In Python, we might provide center + Offset.
# Let's calculate the exact center coords.
u_impulse = M // 2 + 25
v_impulse = N // 2 + 25

# cnotch parameters (type, mode, M, N, C, D0, n)
# C is list of coordinates.
H = cnotch("ideal", "reject", M, N, [[u_impulse, v_impulse]], 2)

# Scale filter for display
Hc = intScaling4e(fftshift(H))  # H is usually centered already by cnotch?
# Typically cnotch returns centered filter H(u,v) corresponding to centered spectrum?
# Wait, MATLAB cnotch returns H matching centered spectrum if we use dftfilt with centered?
# Let's check cnotch/dftfilt implementation.
# Usually: dftfilt(f, H) expects H to be aligned with fft2(f) (uncentered) OR centered.
# Standard dftfilt often handles centering/uncentering.
# If cnotch returns centered H, we might need to ifftshift it before dftfilt if dftfilt expects uncentered.
# Or dftfilt checks.
# I'll check dftfilt.py content.

# Apply filter
gf_raw = dftfilt(g, H)
gf = intScaling4e(gf_raw)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(gs, cmap="gray")
axes[0].set_title("Noisy Image")
axes[0].axis("off")

axes[1].imshow(Glog, cmap="gray")
axes[1].set_title("Spectrum (Log)")
axes[1].axis("off")

axes[2].imshow(Hc, cmap="gray")
axes[2].set_title("Notch Filter")
axes[2].axis("off")

axes[3].imshow(gf, cmap="gray")
axes[3].set_title("Restored Image")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("Figure245.png")
print("Saved Figure245.png")

plt.show()
