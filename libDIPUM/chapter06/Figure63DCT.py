from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from helpers.MyDisp import MyDisp


def dctmtx(N: Any):
    """dctmtx."""
    k = np.arange(N).reshape(-1, 1)
    n = np.arange(N).reshape(1, -1)
    D = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    D[0, :] = D[0, :] / np.sqrt(2)
    return np.sqrt(2 / N) * D


# Parameters
NR = 16
NC = 1

# DCT matrix
DCT = dctmtx(NR)

# Display
plt.figure()
position = 1
Error = MyDisp(DCT, NR, NC, position, "DCT")

# Print to file
plt.savefig("Figure63DCT.png")
plt.show()
