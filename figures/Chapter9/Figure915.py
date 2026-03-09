import numpy as np
import matplotlib.pyplot as plt
import ia870 as ia

# %% SE
Bc4 = ia.iasecross()
Bc8 = ia.iasebox()

# %% Data
X = np.ones((7, 12), dtype=bool)
X[1:3, 4] = False
X[1:3, -2] = False
X = ia.iaintersec(X, np.logical_not(ia.iaframe(X)))

# %% Erosion
Xe4 = ia.iaero(X, Bc4)
Xe8 = ia.iaero(X, Bc8)

# %% Gradient
Grad4 = ia.iasubm(X, Xe4)
Grad8 = ia.iasubm(X, Xe8)

# %% Display
fig = plt.figure(1, figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(X, cmap="gray")
plt.title("X")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(Xe4, cmap="gray")
plt.title(r"Xe4 = $\epsilon_{B_4}(X)$")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(Grad4, cmap="gray")
plt.title(r"$\rho_4(X) = X-Xe4$")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(Xe8, cmap="gray")
plt.title(r"Xe8 = $\epsilon_{B_8}(X)$")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(Grad8, cmap="gray")
plt.title(r"$\rho_8(X) = X-Xe8$")
plt.axis("off")

plt.tight_layout()
fig.savefig("Figure915.png", dpi=150, bbox_inches="tight")
plt.show()
