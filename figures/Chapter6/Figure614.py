import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from libDIP.tmat4e import tmat4e

# %% Figure614

# Parameters
N = 4
Te = 1e-3  # Sampling period

# Data
f = (np.array([0, 1, 2, 3], dtype=float) ** 2).reshape(-1, 1)

# DST Transform (MATLAB dstmtx)
DST = tmat4e('DST', N)

# Reconstruction
t = np.arange(0, 4 * N + Te, Te)
Rec = np.zeros((N, t.size), dtype=float)

Theta = DST @ f

for i in range(1, N + 1):
    Phi = np.sqrt(2.0 / (N + 1)) * np.sin((t + 1.0) * i * np.pi / (N + 1))
    if i == 1:
        Rec[i - 1, :] = Phi * Theta[i - 1, 0]
    else:
        Rec[i - 1, :] = Rec[i - 2, :] + Phi * Theta[i - 1, 0]

Factor = f[-1, 0] / Rec[N - 1, 3000]

# Original digital signal
tn = np.arange(0, N)

# Display
plt.figure(figsize=(8, 8))
for iter_idx in range(1, N + 1):
    ax = plt.subplot(N, 1, iter_idx)
    ax.plot(t, Factor * Rec[iter_idx - 1, :])
    ax.stem(tn, f.ravel(), basefmt=' ')
    ax.set_xlim(t[0], t[-1])

plt.tight_layout()
plt.savefig('Figure614.png')
print('Saved Figure614.png')
plt.show()
