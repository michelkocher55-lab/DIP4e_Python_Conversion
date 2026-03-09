import numpy as np

from libDIPUM.bwboundaries import bwboundaries
from libDIPUM.freemanChainCode import freemanChainCode

print("Running Figure1217 (Freeman chain code)...")

# Parameters
Conn = 4

# Data
X = np.ones((4, 7), dtype=bool)
X[0, 5:7] = False
X[3, [0, 5, 6]] = False

# Boundaries
B_list = bwboundaries(X, Conn)
if not B_list:
    raise RuntimeError("No boundary found.")
B = B_list[0]

# Freeman chain code
c = freemanChainCode(B, Conn)

print("c.x0y0   =", c.x0y0)
print("c.fcc    =", c.fcc)
print("c.mm     =", c.mm)
print("c.diff   =", c.diff)
print("c.diffmm =", c.diffmm)
