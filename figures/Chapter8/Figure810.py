from typing import Any
import numpy as np
import matplotlib.pyplot as plt


# Figure810


def cdf(w: Any, mu: Any, beta: Any):
    """cdf."""
    if w < -5:
        return 0.0
    if w > 5:
        return 1.0
    return 0.5 * (1.0 + (np.sign(w - mu) * (1.0 - np.exp(-abs(w - mu) / beta))))


def one_sided_geometric(select: Any):
    """one_sided_geometric."""
    rho = 0.25
    x = np.ones((10, 3), dtype=float)
    xval = np.linspace(0, 9, 10)

    for j in range(3):
        for i in range(10):
            x[i, j] = (1.0 - rho) * (rho**i)
        rho = rho + 0.25

    z = np.floor(1000 * x[:, select - 1]).astype(int)
    zsum = int(np.sum(z))
    zz = np.zeros(zsum, dtype=int)

    symbol = 0
    index = 0
    for i in range(len(z)):
        for _ in range(z[i]):
            zz[index] = symbol
            index += 1
        symbol += 1

    return x, xval, zz


def two_sided_exp():
    """two_sided_exp."""
    mu = 0.0
    beta = np.sqrt(0.5)
    xval = np.linspace(-5, 5, 11)
    y = np.zeros(11, dtype=float)
    psum = 0.0

    for i in range(11):
        y[i] = cdf(xval[i] + 0.5, mu, beta) - cdf(xval[i] - 0.5, mu, beta)
        psum = psum + y[i]

    return xval, y, psum


def two_sided_exp_reordered(y: Any):
    """two_sided_exp_reordered."""
    xval = np.linspace(0, 9, 10)
    z = xval.copy()
    v = np.array([6, 5, 7, 4, 8, 3, 9, 2, 10, 1], dtype=int)
    for i in range(10):
        z[i] = y[v[i] - 1]
    return z


# Parameters
select = 3

# Process
x, xval, zz = one_sided_geometric(select)
xval1, y, psum = two_sided_exp()
print(psum)
z = two_sided_exp_reordered(y)

# Display
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(xval, x[:, 0], "k-s", xval, x[:, 1], "k--o", xval, x[:, 2], "k:d")
axes[0].set_xlim(0, 9)
axes[0].set_ylim(0, 1)
axes[0].set_xticks(np.arange(0, 10, 1))
axes[0].set_box_aspect(1)
axes[0].set_xlabel("n")
axes[0].set_ylabel("Probability")
axes[0].legend(["0.25", "0.5", "0.75"], frameon=False)
axes[0].set_title("Geometric Distributions")

axes[1].plot(xval1, y, "k-s")
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(0, 1)
axes[1].set_xticks(np.arange(-5, 6, 1))
axes[1].set_box_aspect(1)
axes[1].set_xlabel("x")
axes[1].text(0, 0.9, "mean 0, variance 1")
axes[1].set_title("Laplacian Distribution")

axes[2].plot(xval, z, "k-s")
axes[2].set_xlim(0, 9)
axes[2].set_ylim(0, 1)
axes[2].set_xticks(np.arange(0, 10, 1))
axes[2].set_box_aspect(1)
axes[2].set_xlabel("n")
axes[2].set_title("Interleaved Laplacian Distribution")

plt.tight_layout()
fig.savefig("Figure810.png", dpi=300, bbox_inches="tight")
plt.show()
