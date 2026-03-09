import numpy as np
import matplotlib.pyplot as plt
from libDIP.intScaling4e import intScaling4e

print("Running Figure423 (Spectra of Centered Rectangle)...")

# Data Centered rectangle
f = np.zeros((512, 512))
f[207:304, 247:264] = 1

# DFT
# F=fft2(f);
F = np.fft.fft2(f)

# S=abs(F);
S = np.abs(F)

# Spectrum = intScaling4e(S);
Spectrum = intScaling4e(S)

# Sc=fftshift(S);
Sc = np.fft.fftshift(S)

# SpectrumCentered = intScaling4e(Sc);
SpectrumCentered = intScaling4e(Sc)

# ScLog=log10(1 + abs(Sc));
ScLog = np.log10(1 + np.abs(Sc))

# SpectrumCenteredLog = intScaling4e(ScLog);
SpectrumCenteredLog = intScaling4e(ScLog)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(Spectrum, cmap="gray")
axes[0, 1].set_title("Spectrum (Uncentered)")
axes[0, 1].axis("off")

axes[1, 0].imshow(SpectrumCentered, cmap="gray")
axes[1, 0].set_title("Centered Spectrum")
axes[1, 0].axis("off")

axes[1, 1].imshow(SpectrumCenteredLog, cmap="gray")
axes[1, 1].set_title("Centered Log Spectrum")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure423.png")
print("Saved Figure423.png")
plt.show()
