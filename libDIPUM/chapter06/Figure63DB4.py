import numpy as np
import matplotlib.pyplot as plt

from libDIPUM.wavedec import wavedec
from helpers.MyDisp import MyDisp

# Parameters
NR = 16
NC = 1
LN = int(np.log2(NR))
I = np.eye(NR)

# DWT with Daubechies 4 wavelets.
DB4 = np.zeros((NR, NR))
for i in range(NR):
    DB4[:, i], _ = wavedec(I[:, i], LN, "db4")

# Display
plt.figure()
position = 1
Error = MyDisp(DB4, NR, NC, position, "DFT real")

# Print to file
plt.savefig("Figure63DB4.png")
plt.show()
