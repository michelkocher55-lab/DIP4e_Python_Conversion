import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from libDIP.basisImage4e import basisImage4e

# Parameters
N = 8
P = 1
position = 1
plots = 2
LN = int(np.log2(N))
Te = 1e-3
t = np.arange(1, N + Te, Te)

# Slant transform matrix
a = 3 / np.sqrt(5)
b = 1 / np.sqrt(5)
sp = np.array([
    [1, 1, 1, 1],
    [a, b, -b, -a],
    [1, -1, -1, 1],
    [b, -a, a, -b]
], dtype=float)

for i in range(3, LN + 1):
    NN = 2 ** i
    aN = np.sqrt((3 * NN ** 2) / (4 * (NN ** 2 - 1)))
    bN = np.sqrt((NN ** 2 - 4) / (4 * (NN ** 2 - 1)))

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

    m2 = np.block([
        [sp, np.zeros_like(sp)],
        [np.zeros_like(sp), sp]
    ])

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

S_COMPOSITE, S_DISPLAY = basisImage4e('SLT', N, P)

# Display
plt.figure()
plt.subplot(1, 2, 1)
for i in range(N):
    ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
    markerline, stemlines, baseline = ax.stem(
        np.arange(1, N + 1),
        SLANT[i, :],
        markerfmt='o',
        basefmt=' '
    )
    markerline.set_markeredgecolor('none')
    markerline.set_markerfacecolor((0, 105/255, 166/255))
    markerline.set_markersize(2.25 * 4 / 1.5)
    stemlines.set_linewidth(0.5 * 0.5 / 0.75)

    Temp = interp1d(np.arange(1, N + 1), SLANT[i, :], kind='previous', bounds_error=False,
                    fill_value=(SLANT[i, 0], SLANT[i, -1]))(t)
    ax.plot(t, Temp)

    ax.set_frame_on(False)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if i == 0:
        ax.set_title('SLANT')

plt.subplot(1, 2, 2)
plt.imshow(S_DISPLAY, cmap='gray', vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max())
plt.axis('off')

plt.savefig('Figure617.png')
plt.show()
