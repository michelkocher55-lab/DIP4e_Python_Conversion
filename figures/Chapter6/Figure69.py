import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from libDIP.tmat4e import tmat4e

# %% Figure69

# Parameters
N = 8

# Data
f = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float)

# DHT / DFT matrices
DHT = tmat4e('DHT', N)

# MATLAB dftmtx(N)
k = np.arange(N)
DFT = np.exp(-1j * 2 * np.pi * np.outer(k, k) / N)

# Reconstruction buffers
Rec = np.zeros((N, 8001), dtype=float)
Rec1 = np.zeros((N, 8001), dtype=complex)

t = np.arange(0, N + 1 / 1000.0, 1 / 1000.0)

Theta = DHT @ f
Theta1 = DFT @ f

for i in range(N):
    Phi = np.cos(2 * np.pi * t * i / N) + np.sin(2 * np.pi * t * i / N)
    Phi1 = np.cos(2 * np.pi * t * i / N) + 1j * np.sin(2 * np.pi * t * i / N)

    if i == 0:
        Rec[i, :] = Phi * Theta[i]
        Rec1[i, :] = Phi1 * Theta1[i]
    else:
        Rec[i, :] = Rec[i - 1, :] + Phi * Theta[i]
        Rec1[i, :] = Rec1[i - 1, :] + Phi1 * Theta1[i]

Factor = 1.0 / Rec[7, 0]
Factor1 = 1.0 / Rec1[7, 0]

tn = np.arange(0, 8)

# Display
plt.figure(figsize=(10, 7), dpi=100)
for iter_idx in range(N):
    ax1 = plt.subplot(N, 2, 2 * iter_idx + 1)
    ax1.plot(t, Factor * Rec[iter_idx, :])
    ax1.stem(tn, f, linefmt='C1-', markerfmt='C1o', basefmt=' ')
    ax1.set_xlim(t[0], t[-1])

    ax2 = plt.subplot(N, 2, 2 * iter_idx + 2)
    ax2.plot(t, np.real(Factor1 * Rec1[iter_idx, :]))
    ax2.stem(tn, f, linefmt='C1-', markerfmt='C1o', basefmt=' ')
    ax2.set_xlim(t[0], t[-1])

plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06, wspace=0.18, hspace=0.35)
plt.savefig('Figure69.png')
print('Saved Figure69.png')
plt.show()
