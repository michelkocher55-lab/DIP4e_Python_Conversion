from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from libDIP.basisImage4e import basisImage4e
from libDIP.tmat4e import tmat4e


def MyDisp(Matrix: Any, N: Any, plots: Any, position: Any, Title: Any):
    """MyDisp."""
    Factor = 1.0 / Matrix[0, 0]
    t = np.arange(0, N + 1 / 1000, 1 / 1000.0)

    for i in range(N):
        f = np.cos(2 * np.pi * t * i / N) + np.sin(2 * np.pi * t * i / N)

        ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
        markerline, stemlines, baseline = ax.stem(
            np.arange(0, N), Matrix[i, :], markerfmt="o", basefmt=" "
        )
        markerline.set_markeredgecolor("none")
        markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
        markerline.set_markersize(2.25 * 4 / 1.5)
        stemlines.set_linewidth(0.5 * 0.5 / 0.75)

        ax.plot(t, f / Factor)

        ax.set_frame_on(False)
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 0:
            ax.set_title(Title)


# Parameters
N = 8
P = 1
position = 1
plots = 2

# Process
S_COMPOSITE, S_DISPLAY = basisImage4e("DHT", 8, 1)
DHT = tmat4e("DHT", N)

# Display
plt.figure()
plt.subplot(1, 2, 1)
MyDisp(DHT, N, plots, position, "DHT")
plt.subplot(1, 2, 2)
plt.imshow(S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max())
plt.axis("off")

# Print to file
plt.savefig("Figure68.png")
plt.show()
