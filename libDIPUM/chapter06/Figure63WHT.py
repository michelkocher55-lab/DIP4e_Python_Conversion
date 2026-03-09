import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

from helpers.MyDisp import MyDisp

# Parameters
NR = 16
NC = 1

# Walsh Hadamard transform (normalized)
HAD = hadamard(NR) / np.sqrt(NR)

# Walsh sequency ordering
HadIdx = np.arange(NR)
M = int(np.log2(NR)) + 1

# Bit reverse
binHadIdx = np.array([list(np.binary_repr(i, width=M)) for i in HadIdx], dtype=int)
binHadIdx = np.fliplr(binHadIdx)

binSeqIdx = np.zeros((NR, M - 1), dtype=int)
for k in range(M - 1, 0, -1):
    binSeqIdx[:, k - 1] = np.bitwise_xor(binHadIdx[:, k], binHadIdx[:, k - 1])

SeqIdx = binSeqIdx.dot(2 ** np.arange(M - 2, -1, -1))
WHT = HAD[SeqIdx, :]

# Display
plt.figure()
position = 1
Error = MyDisp(WHT, NR, NC, position, "Walsh Hadamard")

# Print to file
plt.savefig("Figure63WHT.png")
plt.show()
