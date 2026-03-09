import numpy as np
import matplotlib.pyplot as plt

from libDIPUM.wavedec import wavedec
from General.MyDisp import MyDisp

# Parameters
NR = 16
NC = 2
LN = int(np.log2(NR))
I = np.eye(NR)

# Biorthogonal wavelets
BIOR31 = np.zeros((NR, NR))
RBIO31 = np.zeros((NR, NR))

for i in range(NR):
    BIOR31[:, i], _ = wavedec(I[:, i], LN, "bior3.1")
    RBIO31[:, i], _ = wavedec(I[:, i], LN, "rbio3.1")

# Display
plt.figure()
position = 1
Error_BIOR31 = MyDisp(BIOR31, NR, NC, position, "DFT real")
position = 2
Error_RBIO31 = MyDisp(RBIO31, NR, NC, position, "DFT real")

# Print to file
plt.savefig("Figure63BIOR31.png")
plt.show()
