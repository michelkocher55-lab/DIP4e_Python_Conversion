import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.lpfilter import lpfilter
from libDIP.intScaling4e import intScaling4e

# Parameters
D0 = 5
n_vals = [1, 2, 5, 20]

hs1 = []
profiles = []

for n in n_vals:
    H = lpfilter("butterworth", 1000, 1000, D0, n)
    M, N = H.shape
    h1 = np.real(np.fft.fftshift(np.fft.ifft2(H)))
    hs1.append(intScaling4e(h1))
    profiles.append(h1[M // 2, :])

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
plt.savefig("Figure447.png")
plt.show()
