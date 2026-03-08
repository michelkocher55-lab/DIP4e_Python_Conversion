import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from libDIP.basisImage4e import basisImage4e
from libDIP.tmat4e import tmat4e


def MyDisp(Matrix, N, plots, position, Title):
    Factor = 1.0 / Matrix[0, 0]
    t = np.arange(0, N + 1 / 1000.0, 1 / 1000.0)

    for i in range(1, N + 1):
        f = np.sqrt(2.0 / (N + 1)) * np.sin((t + 1.0) * i * np.pi / (N + 1))

        ax = plt.subplot(N, plots, (plots * i) - (plots - position))
        markerline, stemlines, baseline = ax.stem(
            np.arange(0, N),
            Matrix[i - 1, :],
            markerfmt='o',
            basefmt=' '
        )

        markerline.set_markeredgecolor('none')
        markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
        markerline.set_markersize(2.25 * 4 / 1.5)

        # MATLAB sets stem line color to none; emulate by hiding stem lines.
        stemlines.set_color('none')
        stemlines.set_linewidth(0.5 * 0.5 / 0.75)

        ax.plot(t, f)

        ax.set_frame_on(False)
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 1:
            ax.set_title(Title)


# %% Figure 613

# Parameters
N = 8
P = 1
position = 1
plots = 2

# Process
S_COMPOSITE, S_DISPLAY = basisImage4e('DST', 8, 1)
DST = tmat4e('DST', N)

# Display
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
MyDisp(DST, N, plots, position, 'DST')

plt.subplot(1, 2, 2)
plt.imshow(S_DISPLAY, cmap='gray', vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max())
plt.axis('off')

# Print to file
plt.savefig('Figure613.png')
print('Saved Figure613.png')
plt.show()
