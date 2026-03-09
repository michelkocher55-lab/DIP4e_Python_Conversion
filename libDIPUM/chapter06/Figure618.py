import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from libDIPUM.wavedec import wavedec
from libDIP.basisImage4e import basisImage4e

# Parameters
N = 8
P = 1
position = 1
plots = 2
LN = int(np.log2(N))
I = np.eye(N)
Te = 1e-3
t = np.arange(1, N + Te, Te)

# Compute Haar matrix
HAAR = np.zeros((N, N))
for i in range(N):
    HAAR[:, i], _ = wavedec(I[:, i], LN, "haar")

S_COMPOSITE, S_DISPLAY = basisImage4e("HAAR", N, P)

# Display
plt.figure()
plt.subplot(1, 2, 1)
for i in range(N):
    ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
    markerline, stemlines, baseline = ax.stem(
        np.arange(1, N + 1), HAAR[i, :], markerfmt="o", basefmt=" "
    )
    markerline.set_markeredgecolor("none")
    markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
    markerline.set_markersize(2.25 * 4 / 1.5)
    stemlines.set_linewidth(0.5 * 0.5 / 0.75)

    Temp = interp1d(
        np.arange(1, N + 1),
        HAAR[i, :],
        kind="previous",
        bounds_error=False,
        fill_value=(HAAR[i, 0], HAAR[i, -1]),
    )(t)
    ax.plot(t, Temp)

    ax.set_frame_on(False)
    ax.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if i == 0:
        ax.set_title("Haar")

plt.subplot(1, 2, 2)
plt.imshow(S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max())
plt.axis("off")

plt.savefig("Figure618.png")
plt.show()
