import numpy as np
import matplotlib.pyplot as plt

from libDIPUM.wavedec import wavedec
from General.MyDisp import MyDisp

# Parameters
NR = 16
NC = 1
LN = int(np.log2(NR))

# DWT with Haar wavelets.
I = np.eye(NR)
HAAR = np.zeros((NR, NR))
for i in range(NR):
    HAAR[:, i], _ = wavedec(I[:, i], LN, "haar")

# Display
plt.figure()
position = 1
Error = MyDisp(HAAR, NR, NC, position, "Haar")

# Print to file
plt.savefig("Figure63HAAR.png")
plt.show()
