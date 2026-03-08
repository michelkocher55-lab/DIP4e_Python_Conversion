import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import ia870 as ia
from General.mmshow import mmshow

# %% Figure927
# Pruning

# %% Parameters
NR = 3
NC = 8

# %% Data
X = np.zeros((12, 16), dtype=bool)
X[1, 6:9] = True
X[2, [5, 9]] = True
X[3, [1, 4]] = True
X[4, [1, 3, 10]] = True
X[5, [1, 2, 10, 11]] = True
X[6, [2, 9, 11]] = True
X[7, [3, 9, 11]] = True
X[8, [2, 9, 11]] = True
X[9, [2, 8, 11, 14]] = True
X[10, np.r_[3:8, 12:14]] = True

# %% Pruning
BFG0 = ia.iaimg2se(np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=bool))
BBG0 = ia.iaimg2se(np.array([[0, 0, 0],
                             [1, 0, 1],
                             [1, 1, 1]], dtype=bool))

Prune = X.copy()
for Iter in range(1, 4):
    BFG = BFG0.copy()
    BBG = BBG0.copy()

    for cpt in range(1, 9):
        if cpt == 1:
            BFG = ia.iaimg2se(np.array([[0, 0, 0],
                                        [1, 1, 0],
                                        [0, 0, 0]], dtype=bool))
            BBG = ia.iaimg2se(np.array([[0, 1, 1],
                                        [0, 0, 1],
                                        [0, 1, 1]], dtype=bool))
        elif cpt == 5:
            BFG = ia.iaimg2se(np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]], dtype=bool))
            BBG = ia.iaimg2se(np.array([[0, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 1]], dtype=bool))

        HMT = ia.iasupgen(Prune, ia.iase2hmt(BFG, BBG))
        Prune = ia.iasubm(Prune, HMT)

        BFG = ia.iaserot(BFG, 90, 'CLOCKWISE')
        BBG = ia.iaserot(BBG, 90, 'CLOCKWISE')

# %% Endpoints detection
EndPoints = np.zeros_like(X, dtype=bool)
for cpt in range(1, 9):
    if cpt == 1:
        BFG = ia.iaimg2se(np.array([[0, 0, 0],
                                    [1, 1, 0],
                                    [0, 0, 0]], dtype=bool))
        BBG = ia.iaimg2se(np.array([[0, 1, 1],
                                    [0, 0, 1],
                                    [0, 1, 1]], dtype=bool))
    elif cpt == 5:
        BFG = ia.iaimg2se(np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]], dtype=bool))
        BBG = ia.iaimg2se(np.array([[0, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=bool))

    EndPoints = ia.iaaddm(EndPoints, ia.iasupgen(Prune, ia.iase2hmt(BFG, BBG)))

    BFG = ia.iaserot(BFG, 90, 'CLOCKWISE')
    BBG = ia.iaserot(BBG, 90, 'CLOCKWISE')

# %% Conditional dilation
CondDil = EndPoints.copy()
for cpt in range(1, 4):
    CondDil = ia.iaintersec(X, ia.iadil(CondDil, ia.iasebox(1)))

# %% Final union
Final = ia.iaunion(CondDil, Prune)

# %% Display
fig = plt.figure(1, figsize=(12, 8))
try:
    fig.canvas.manager.set_window_title('Figure 9.27')
except Exception:
    pass

plt.subplot(2, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('X')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(Prune, cmap='gray')
plt.title('Prune = Thin(3, X)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(EndPoints, cmap='gray')
plt.title('EndPoints = HMT(Prune, E)')
plt.axis('off')

plt.subplot(2, 3, 4)
mmshow(CondDil, EndPoints)
plt.title('CondDil = delta^3_X(EndPoints)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(Final, cmap='gray')
plt.title('Final = Prune U CondDil')
plt.axis('off')

plt.tight_layout()
fig.savefig('Figure927.png', dpi=150, bbox_inches='tight')
plt.show()
