import numpy as np
import matplotlib.pyplot as plt

from General.MyDisp import MyDisp

# Parameters
NR = 16
NC = 1

I = np.eye(NR)

# Display
plt.figure()
position = 1
Error = MyDisp(I, NR, NC, position, 'Canonical Basis')

# Print to file
plt.savefig('Figure63STD.png')
plt.show()
