from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from helpers.MyDisp import MyDisp


def dftmtx(N: Any):
    """dftmtx."""
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)
    return W


# Parameters
NR = 16
NC = 2

# DFT matrix (normalized)
DFT = dftmtx(NR) / np.sqrt(NR)

# Display
plt.figure()
position = 1
Error_RealDFT = MyDisp(np.real(DFT), NR, NC, position, "DFT real")
position = 2
Error_ImagDFT = MyDisp(np.imag(DFT), NR, NC, position, "DFT real")

# Print to file
plt.savefig("Figure63DFT.png")
plt.show()
