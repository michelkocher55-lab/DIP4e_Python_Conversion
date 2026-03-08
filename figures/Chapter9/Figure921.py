import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import ia870 as ia
from General.mmshow import mmshow

# %% Figure921

# %% Init
Fig = 1
try:
    _raw = input('Figure921 (1) or Problem921c (2) : ').strip()
    Choice = int(_raw) if _raw else 1
except Exception:
    Choice = 1

# %% Data
if Choice == 1:
    X = np.zeros((12, 11), dtype=bool)
    X[1, 3:6] = True
    X[2, 2:7] = True
    X[3, 2:6] = True
    X[4, 4:6] = True
    X[4, 7] = True
    X[5, 4:6] = True
    X[5, 7] = True
    X[6, 4:7] = True
    X[7, 6] = True
    X[8, 6] = True
    X[9, 5:7] = True
    X[10, 5] = True

    XX = np.zeros((14, 13), dtype=bool)
    XX[1:-1, 1:-1] = X
elif Choice == 2:
    XX = np.zeros((13, 13), dtype=bool)
    XX[2:11, 5:8] = True
    XX[5:8, 2:11] = True
else:
    raise ValueError('Plouc')

NIter = 10

# %% Interval
BFG = np.array([[1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]], dtype=bool)
BBG = np.array([[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]], dtype=bool)

B1 = ia.iase2hmt(ia.iaimg2se(BFG), ia.iaimg2se(BBG))
B2 = ia.iainterot(B1, 90)
B3 = ia.iainterot(B2, 90)
B4 = ia.iainterot(B3, 90)

# Optional interval visualization (as in MATLAB script)
ia.iaintershow(B1)
ia.iaintershow(B2)
ia.iaintershow(B3)
ia.iaintershow(B4)

B = [B1, B2, B3, B4]
NB = len(B)

# %% Convex hull
Frame = ia.iaframe(XX)
ConvexHull = np.zeros_like(XX, dtype=bool)

Y_all = []
HMT_all = []
MaxIter = []

for interval_idx in range(NB):
    y_series = [XX.copy()]           # Y(:,:,Interval,1)
    h_series = []                    # HMT(:,:,Interval,iter-1)

    OK = True
    iter_idx = 2
    while OK:
        h = ia.iasupgen(y_series[iter_idx - 2], B[interval_idx])
        y = ia.iaunion(y_series[iter_idx - 2], h)
        y = ia.iaintersec(y, ia.ianeg(Frame))

        h_series.append(h)
        y_series.append(y)

        if np.any(y_series[iter_idx - 2] != y):
            iter_idx += 1
            if iter_idx > NIter + 2:
                # Safety cap; MATLAB version expects convergence before this.
                OK = False
                MaxIter.append(iter_idx)
        else:
            OK = False
            MaxIter.append(iter_idx)

    Y_all.append(y_series)
    HMT_all.append(h_series)
    ConvexHull = ia.iaunion(ConvexHull, y_series[-1])

# %% Display
saved_figs = []

for interval_idx in range(4):
    fig = plt.figure(Fig, figsize=(12, 6))
    saved_figs.append(fig)
    Fig += 1

    plt.subplot(2, 5, 1)
    plt.imshow(Y_all[interval_idx][0], cmap='gray')
    plt.title('X')
    plt.axis('off')

    max_it = MaxIter[interval_idx]
    # MATLAB: for iter = 2 : MaxIter(Interval)-1
    for iter_disp in range(2, max_it):
        sp_idx = iter_disp
        if sp_idx > 10:
            break
        plt.subplot(2, 5, sp_idx)

        y_disp = Y_all[interval_idx][iter_disp - 1]
        h_disp = HMT_all[interval_idx][iter_disp - 2]
        mmshow(y_disp, h_disp)
        plt.title(f'X^{{I={interval_idx + 1}}}_{{iter={iter_disp}}}')
        plt.axis('off')

    plt.tight_layout()

fig = plt.figure(Fig, figsize=(12, 8))
saved_figs.append(fig)
Fig += 1

plt.subplot(2, 3, 1)
plt.imshow(XX, cmap='gray')
plt.title('X')
plt.axis('off')

for interval_idx in range(4):
    plt.subplot(2, 3, interval_idx + 2)
    plt.imshow(Y_all[interval_idx][-1], cmap='gray')
    plt.title(f'Y^{{I={interval_idx + 1}}}_{{iter = {MaxIter[interval_idx]}}}')
    plt.axis('off')

plt.subplot(2, 3, 6)
mmshow(ConvexHull, XX)
plt.title('Convex hull, original data')
plt.axis('off')

plt.tight_layout()

# %% Print 2 file
for iter_idx in range(1, 6):
    if iter_idx - 1 < len(saved_figs):
        saved_figs[iter_idx - 1].savefig(f'Figure921_{iter_idx}.png', dpi=150, bbox_inches='tight')

plt.show()
