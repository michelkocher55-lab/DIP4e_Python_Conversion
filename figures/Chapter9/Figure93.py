import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from General.mmshow import mmshow
import ia870 as ia

# %% Figure93

# %% Data
X = np.ones((7, 15), dtype=bool)
X[0, :] = False
X[-1, :] = False
X[:, 0] = False
X[:, -1] = False
X[1, 5:10] = False      # MATLAB: X(2, 6:10) = false
X[-2, 5:10] = False     # MATLAB: X(end-1, 6:10) = false

# %% SE
B = ia.iasebox()

# %% Erosion
Y = ia.iaero(X, B)

# %% Display
fig = plt.figure(1, figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('X')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Y, cmap='gray')
plt.title('Y')
plt.axis('off')

plt.subplot(1, 3, 3)
mmshow(X, Y)
plt.title('X, Y')
plt.axis('off')

plt.tight_layout()
fig.savefig('Figure93.png', dpi=150, bbox_inches='tight')
plt.show()
