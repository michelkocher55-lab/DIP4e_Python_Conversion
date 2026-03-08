import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 800
W = 120
C0 = 200

# Process
D = np.arange(0, M)

# lowpass 1
HL1 = np.exp(-(D ** 2 / (W ** 2)))

# lowpass 2
HL2 = np.exp(-(D ** 2 / (4 * W ** 2)))

# highpass from lowpass
Hhigh = 1 - HL2

# Bandreject formed by sum of lowpass and highpass Gaussian filters
H = HL1 + Hhigh

# highpass with shifted 0-point to C0
HhighS = 1 - np.exp(-((D - C0) ** 2) / (W ** 2))

# Formula in book (avoid divide by zero at D=0)
with np.errstate(divide='ignore', invalid='ignore'):
    Hbook = 1 - np.exp(-(((D ** 2 - C0 ** 2) / (D * W)) ** 2))

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(H)
axes[0].set_box_aspect(1)
axes[0].set_title('exp(-(D.^2/(W^2))) + (1 - exp(-(D.^2/(4*W^2))))')
axes[0].axvline(C0, color='black')

axes[1].plot(HhighS)
axes[1].set_box_aspect(1)
axes[1].set_title('(1 - exp(-(D - C0).^2/W^2))')
axes[1].axvline(C0, color='black')

axes[2].plot(Hbook)
axes[2].set_box_aspect(1)
axes[2].set_title('1 - exp(-((D.^2 - C0^2)./(D*W)).^2)')
axes[2].axvline(C0, color='black')

plt.tight_layout()
plt.savefig('Figure461.png')
plt.show()
