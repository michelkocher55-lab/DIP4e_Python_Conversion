import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from scipy.interpolate import interp1d

from libDIP.basisImage4e import basisImage4e

# Parameters
N = 8
P = 1
position = 1
plots = 2
Te = 1e-3
t = np.arange(1, N + Te, Te)

# Walsh Hadamard Matrix (sequency ordered, like MATLAB whtmtx)
HAD = hadamard(N)
HadIdx = np.arange(N)
M = int(np.log2(N)) + 1
binHadIdx = np.array([list(np.binary_repr(i, width=M)) for i in HadIdx], dtype=int)
binHadIdx = np.fliplr(binHadIdx)
binSeqIdx = np.zeros((N, M - 1), dtype=int)
for k in range(M - 1, 0, -1):
    binSeqIdx[:, k - 1] = np.bitwise_xor(binHadIdx[:, k], binHadIdx[:, k - 1])
SeqIdx = binSeqIdx.dot(2 ** np.arange(M - 2, -1, -1))
WHT = HAD[SeqIdx, :]

S_COMPOSITE, S_DISPLAY = basisImage4e('WHT', N, P)

# Display
plt.figure()
plt.subplot(1, 2, 1)
for i in range(N):
    ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
    markerline, stemlines, baseline = ax.stem(
        np.arange(1, N + 1),
        WHT[i, :],
        markerfmt='o',
        basefmt=' '
    )
    markerline.set_markeredgecolor('none')
    markerline.set_markerfacecolor((0, 105/255, 166/255))
    markerline.set_markersize(2.25 * 4 / 1.5)
    stemlines.set_linewidth(0.5 * 0.5 / 0.75)

    Temp = interp1d(np.arange(1, N + 1), WHT[i, :], kind='previous', bounds_error=False, fill_value=(WHT[i, 0], WHT[i, -1]))(t)
    ax.plot(t, Temp)

    ax.set_frame_on(False)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if i == 0:
        ax.set_title('DCT')

plt.subplot(1, 2, 2)
plt.imshow(S_DISPLAY, cmap='gray', vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max())
plt.axis('off')

# Print to file
plt.savefig('Figure616.png')
plt.show()
