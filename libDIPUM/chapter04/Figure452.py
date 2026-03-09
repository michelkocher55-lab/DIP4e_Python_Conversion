import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.lpfilter import lpfilter
from libDIP.intScaling4e import intScaling4e

# Parameters
M = 1000
N = 1000
D0 = 5
n_vals = [1, 2, 5, 20]

# Constant in the frequency domain. Its inverse will be an impulse.
G = np.ones((M, N))
imp = np.real(np.fft.fftshift(np.fft.ifft2(G)))

hs1 = []
profiles = []

for n in n_vals:
    HLP = lpfilter("butterworth", M, N, D0, n)
    hLP = np.real(np.fft.fftshift(np.fft.ifft2(HLP)))
    hHP = imp - hLP / np.max(hLP)

    hs1.append(intScaling4e(hHP))
    profiles.append(hHP[M // 2, :])

profiles = np.vstack(profiles).T

# Display
fig = plt.figure(figsize=(12, 6))

for idx, n in enumerate(n_vals):
    ax = fig.add_subplot(2, 4, idx + 1)
    ax.imshow(hs1[idx], cmap="gray")
    ax.axis("off")

    axp = fig.add_subplot(2, 4, idx + 5)
    axp.plot(profiles[:, idx])
    # axp.set_aspect('equal', adjustable='box')
    axp.axis("off")

plt.tight_layout()
plt.savefig("Figure452.png")
plt.show()
