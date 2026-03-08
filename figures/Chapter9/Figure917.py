import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import ia870 as ia

# %% Figure917
# Hole filling

# %% Parameters
R = 3
C = 3

# %% Data
f = np.zeros((10, 7), dtype=bool)
f[1, 2:4] = True
f[2:4, [1, 4]] = True
f[4:6, [2, 4]] = True
f[6:8, [1, 5]] = True
f[8, 1:5] = True

# %% SE
B = ia.iasecross(1)

# %% Manual hole filling
X_list = []

x0 = np.zeros_like(f, dtype=bool)
x0[R - 1, C - 1] = True
X_list.append(x0)

x1 = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[0], B))
X_list.append(x1)

while True:
    xk = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[-1], B))
    X_list.append(xk)
    if np.array_equal(X_list[-1], X_list[-2]):
        break

g = ia.iaunion(f, X_list[-1])

# %% Display
fig = plt.figure(1, figsize=(12, 9))

plt.subplot(3, 4, 1)
plt.imshow(f, cmap='gray')
plt.title('A')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(ia.ianeg(f), cmap='gray')
plt.title('A^C')
plt.axis('off')

for iter_idx in range(len(X_list)):
    plt.subplot(3, 4, iter_idx + 3)
    plt.imshow(X_list[iter_idx], cmap='gray')
    plt.title(f'X_{{{iter_idx}}}')
    plt.axis('off')

plt.tight_layout()
fig.savefig('Figure917.png', dpi=150, bbox_inches='tight')
plt.show()
