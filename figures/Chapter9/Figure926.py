import numpy as np
import matplotlib.pyplot as plt
import ia870 as ia

# %% Figure926
# Maximum ball skeleton
# Reconstruction using a non homotopic skeleton

# %% Init
Fig = 1

# %% Data
X = np.ones((12, 7), dtype=bool)
X[0, :] = False
X[-1, :] = False
X[:, 0] = False
X[:, -1] = False

X[2:6, 1] = False
X[1, 2:6] = False
X[2, 4:6] = False
X[3, 4:6] = False
X[4:6, 5] = False

# %% Structuring element
B = ia.iasebox(1)

# %% Process
levels = 3  # k = 0..2
nr, nc = X.shape

Ero = np.zeros((nr, nc, levels), dtype=bool)
Open = np.zeros((nr, nc, levels), dtype=bool)
IndividualSkeleton = np.zeros((nr, nc, levels), dtype=bool)
Skeleton = np.zeros((nr, nc, levels), dtype=bool)
DilIndividualSkeleton = np.zeros((nr, nc, levels), dtype=bool)
ReconstructedImage = np.zeros((nr, nc, levels), dtype=bool)

for k in range(levels):
    Ero[:, :, k] = ia.iaero(X, ia.iasebox(k))
    Open[:, :, k] = ia.iaopen(Ero[:, :, k], B)
    IndividualSkeleton[:, :, k] = ia.iasubm(Ero[:, :, k], Open[:, :, k])

    if k == 0:
        Skeleton[:, :, k] = IndividualSkeleton[:, :, k]
    else:
        Skeleton[:, :, k] = ia.iaaddm(
            Skeleton[:, :, k - 1], IndividualSkeleton[:, :, k]
        )

    DilIndividualSkeleton[:, :, k] = ia.iadil(
        IndividualSkeleton[:, :, k], ia.iasebox(k)
    )

    if k == 0:
        ReconstructedImage[:, :, k] = DilIndividualSkeleton[:, :, k]
    else:
        ReconstructedImage[:, :, k] = ia.iaaddm(
            ReconstructedImage[:, :, k - 1], DilIndividualSkeleton[:, :, k]
        )

# %% Display
fig = plt.figure(Fig, figsize=(18, 9))

for k in range(levels):
    plt.subplot(3, 6, 1 + 6 * k)
    plt.imshow(Ero[:, :, k], cmap="gray")
    plt.title(f"ero(A, {k}B)")
    plt.axis("off")

    plt.subplot(3, 6, 2 + 6 * k)
    plt.imshow(Open[:, :, k], cmap="gray")
    plt.title(f"open(ero(A, {k}B), B)")
    plt.axis("off")

    plt.subplot(3, 6, 3 + 6 * k)
    plt.imshow(IndividualSkeleton[:, :, k], cmap="gray")
    plt.title(f"S_{k}(A)")
    plt.axis("off")

    plt.subplot(3, 6, 4 + 6 * k)
    plt.imshow(Skeleton[:, :, k], cmap="gray")
    plt.title("union(S_k(A))")
    plt.axis("off")

    plt.subplot(3, 6, 5 + 6 * k)
    plt.imshow(DilIndividualSkeleton[:, :, k], cmap="gray")
    plt.title(f"S_{k}(A) dil {k}B")
    plt.axis("off")

    plt.subplot(3, 6, 6 + 6 * k)
    plt.imshow(ReconstructedImage[:, :, k], cmap="gray")
    plt.title(f"union(S_{k}(A) dil {k}B)")
    plt.axis("off")

plt.tight_layout()
fig.savefig("Figure926.png", dpi=150, bbox_inches="tight")
plt.show()
