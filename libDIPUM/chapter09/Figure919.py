import numpy as np
import matplotlib.pyplot as plt
import ia870 as ia

# %% Figure919

# %% SE
Bc4 = ia.iasecross()
Bc8 = ia.iasebox()

# %% Data
X = np.zeros((10, 10), dtype=bool)
X[1:3, 6:9] = True
X[1:3, -2] = True
X[3, [5, 6, 8]] = True
X[4, 5:9] = True
X[5, 3:6] = True
X[6, 2:5] = True
X[7, [1, 4]] = True
X[8, 2:5] = True

# %% Conditional dilation
LesX = np.zeros((10, 10, 7), dtype=bool)
LesX[6, 3, 0] = True
for iter_idx in range(1, 7):
    LesX[:, :, iter_idx] = ia.iaintersec(
        ia.iadil(LesX[:, :, iter_idx - 1], ia.iasebox()), X
    )

# %% Display
fig = plt.figure(1, figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.imshow(X, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(LesX[:, :, 0], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(LesX[:, :, 1], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(LesX[:, :, 2], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(LesX[:, :, 3], cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(LesX[:, :, 6], cmap="gray")
plt.axis("off")

plt.tight_layout()
fig.savefig("Figure919.png", dpi=150, bbox_inches="tight")
plt.show()
