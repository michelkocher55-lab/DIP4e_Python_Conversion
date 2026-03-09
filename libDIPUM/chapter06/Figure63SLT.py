import numpy as np
import matplotlib.pyplot as plt

from helpers.MyDisp import MyDisp

# Parameters
N = 16
NC = 1
LN = int(np.log2(N))

# Slant transform
a = 3 / np.sqrt(5)
b = 1 / np.sqrt(5)
sp = np.array(
    [[1, 1, 1, 1], [a, b, -b, -a], [1, -1, -1, 1], [b, -a, a, -b]], dtype=float
)

for i in range(3, LN + 1):
    NN = 2**i
    aN = np.sqrt((3 * NN**2) / (4 * (NN**2 - 1)))
    bN = np.sqrt((NN**2 - 4) / (4 * (NN**2 - 1)))

    sr1 = np.array([[1, 0], [aN, bN]], dtype=float)
    sr2 = np.array([[1, 0], [-aN, bN]], dtype=float)
    sz = np.zeros((2, (NN - 4) // 2))
    sn1 = np.hstack([sr1, sz, sr2, sz])

    q = (NN // 2) - 2
    ir = np.eye(q)
    iz = np.zeros((q, 2))
    sn2 = np.hstack([iz, ir, iz, ir])
    sn4 = np.hstack([iz, ir, iz, -ir])

    sr1 = np.array([[0, 1], [-bN, aN]], dtype=float)
    sr2 = np.array([[0, -1], [bN, aN]], dtype=float)
    sn3 = np.hstack([sr1, sz, sr2, sz])

    sn = np.vstack([sn1, sn2, sn3, sn4])

    m2 = np.block([[sp, np.zeros_like(sp)], [np.zeros_like(sp), sp]])

    sp = sn @ m2

    SLANT = np.zeros_like(sp)
    for k in range(NN):
        if k < 2:
            seq = k
        elif k <= NN // 2 - 1:
            if k % 2 == 0:
                seq = 2 * k
            else:
                seq = 2 * k + 1
        elif k == NN // 2:
            seq = 2
        elif k == NN // 2 + 1:
            seq = 3
        else:
            if k % 2 == 0:
                seq = 2 * (k - NN // 2) + 1
            else:
                seq = 2 * (k - NN // 2)
        SLANT[seq, :] = sp[k, :]
    sp = SLANT

sp = sp / np.sqrt(N)
SLANT = sp.copy()

# Display
plt.figure()
position = 1
Error = MyDisp(SLANT, N, NC, position, "Slant")

# Print to file
plt.savefig("Figure63SLT.png")
plt.show()
