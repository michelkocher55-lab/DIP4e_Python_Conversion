import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import ia870 as ia
from General.mmshow import mmshow

# %% Figure914

# %% Data
X = np.zeros((7, 15), dtype=bool)
X[1:6, 1:5] = True
X[1:6, -5:-1] = True
X[2:5, 5:10] = True
X[3, 5] = False
X[3, 9] = False

# %% Interval
I1FG = np.ones((3, 3), dtype=bool)
I1FG[1, 1] = False
I1BG = np.logical_not(I1FG)
I1 = ia.iase2hmt(ia.iaimg2se(I1FG), ia.iaimg2se(I1BG))
ia.iaintershow(I1)

I2FG = np.zeros((3, 3), dtype=bool)
I2FG[1:, 0:2] = True
I2BG = np.logical_not(I2FG)
I2 = ia.iase2hmt(ia.iaimg2se(I2FG), ia.iaimg2se(I2BG))
ia.iaintershow(I2)

I3FG = np.array([
    [False, False, False],
    [True,  True,  False],
    [False, False, False],
], dtype=bool)
I3BG = np.array([
    [False, False, True],
    [False, False, True],
    [False, False, True],
], dtype=bool)
I3 = ia.iase2hmt(ia.iaimg2se(I3FG), ia.iaimg2se(I3BG))
ia.iaintershow(I3)

# %% HMT
Y1 = ia.iasupgen(X, I1)
Y2 = ia.iasupgen(X, I2)
Y3 = ia.iasupgen(X, I3)

# %% Display
fig = plt.figure(1, figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('X')
plt.axis('off')

plt.subplot(2, 2, 2)
mmshow(X, Y1)
plt.title('HMT(X, I1)')
plt.axis('off')

plt.subplot(2, 2, 3)
mmshow(X, Y2)
plt.title('HMT(X, I2)')
plt.axis('off')

plt.subplot(2, 2, 4)
mmshow(X, Y3)
plt.title('HMT(X, I3)')
plt.axis('off')

plt.tight_layout()
fig.savefig('Figure914.png', dpi=150, bbox_inches='tight')
plt.show()
