import numpy as np

from libDIPUM.bwboundaries import bwboundaries
from libDIPUM.freemanChainCode import freemanChainCode

print('Running Figure1216 (Freeman chain code examples)...')

# Parameters
Conn = 4

# Data
X = []
X.append(np.ones((2, 2), dtype=bool))
X.append(np.ones((2, 3), dtype=bool))
X.append(np.ones((3, 3), dtype=bool))
x4 = np.ones((3, 3), dtype=bool)
x4[0, -1] = False
X.append(x4)
X.append(np.ones((2, 4), dtype=bool))

# Boundaries and Freeman chain code
for iter_idx, Xi in enumerate(X, start=1):
    B_list = bwboundaries(Xi, conn=Conn)
    if not B_list:
        print(f'iter {iter_idx}: no boundary found')
        continue

    B = B_list[0]
    c = freemanChainCode(B, Conn)

    # MATLAB prints `c` in the loop; print all key fields explicitly.
    print(f'iter {iter_idx}:')
    print('  x0y0   =', c.x0y0)
    print('  fcc    =', c.fcc)
    print('  mm     =', c.mm)
    print('  diff   =', c.diff)
    print('  diffmm =', c.diffmm)
