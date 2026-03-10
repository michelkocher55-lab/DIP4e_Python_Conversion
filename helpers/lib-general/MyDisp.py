from typing import Any
import numpy as np
import matplotlib.pyplot as plt


def MyDisp(Matrix: Any, N: Any, plots: Any, position: Any, Title: Any):
    """MyDisp."""
    I = np.eye(N)

    for i in range(N):
        ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
        markerline, stemlines, baseline = ax.stem(
            np.arange(1, N + 1), Matrix[i, :], markerfmt="o", basefmt=" "
        )
        markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
        markerline.set_markeredgecolor((0, 105 / 255, 166 / 255))
        markerline.set_markersize(2.0)
        stemlines.set_linewidth(0.5 * 0.5 / 0.75)

        ax.set_frame_on(False)
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if i == 0:
            ax.set_title(Title)

    Error = np.sum((Matrix @ Matrix.conj().T) - I)
    return Error
