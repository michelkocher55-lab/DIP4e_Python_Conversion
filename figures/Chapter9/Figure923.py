import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import ia870 as ia

# %% Figure923

# %% Init
Fig = 1
NR = 3
NC = 8
SDCCompare = 1

# %% Data
Choix = 1
if Choix == 1:
    X = np.ones((7, 13), dtype=bool)
    X[0, :] = False
    X[-1, :] = False
    X[:, 0] = False
    X[:, -1] = False
    X[-2, 4:6] = False
    X[2:, -3:-1] = False
elif Choix == 2:
    X = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
    ], dtype=bool)
else:
    raise ValueError('Plouc')

# %% SDC Manual Thinning
fig1 = plt.figure(Fig, figsize=(16, 6))
Fig += 1

Y1 = X.copy()
Iter1 = 1
while True:
    YOld1 = Y1.copy()

    L = ia.iahomothin()
    ia.iaintershow(L)

    for cpt in range(1, 9):
        HMT1 = ia.iasupgen(Y1, L)
        Y1 = ia.iasubm(Y1, HMT1)

        subplot_idx = cpt + (Iter1 - 1) * NC
        if subplot_idx <= NR * NC:
            plt.subplot(NR, NC, subplot_idx)
            plt.imshow(Y1, cmap='gray')
            plt.axis('off')
            plt.title(f'theta={45 * (cpt - 1)}, iter={Iter1}')

        L = ia.iainterot(L, 45)
        ia.iaintershow(L)

    if np.array_equal(YOld1, Y1):
        break
    else:
        Iter1 += 1

# %% SDC All in one thinning
Y2 = ia.iathin(X, ia.iahomothin(), -1, 45, 'CLOCKWISE')

# %% Check
if SDCCompare:
    OK2 = np.array_equal(Y1, Y2)
    print(f'OK2 = {OK2}')

# %% Final display
fig2 = plt.figure(Fig, figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('X')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Y1, cmap='gray')
plt.title('Y')
plt.axis('off')

plt.tight_layout()

# %% Print
fig1.savefig('Figure923.png', dpi=150, bbox_inches='tight')
fig2.savefig('Figure923Bis.png', dpi=150, bbox_inches='tight')
plt.show()
