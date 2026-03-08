"""Figure 13.20 - Minimum distance / Gaussian classifier in 3D."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


print("Running Figure1320...")
np.set_printoptions(precision=3, suppress=True)

# Data
try:
    choix_txt = input("From the book (1) more numbers (2) : ").strip()
except EOFError:
    choix_txt = "1"

choix = 1 if choix_txt == "" else int(choix_txt)

if choix == 1:
    X1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=float,
    ).T  # (3,4)

    X2 = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=float,
    ).T  # (3,4)
elif choix == 2:
    X1 = 0.2 * np.random.randn(3, 10) + np.tile(np.array([[1.0], [1.0], [1.0]]), (1, 10))
    X2 = 0.2 * np.random.randn(3, 10) + np.tile(np.array([[-1.0], [-1.0], [-1.0]]), (1, 10))
else:
    raise ValueError("Plouc")

N1 = X1.shape[1]
N2 = X2.shape[1]

# Mean and covariance (matching MATLAB comments/formulas)
# C = X*X'/N - m*m'
m1 = np.mean(X1, axis=1, keepdims=True)
m2 = np.mean(X2, axis=1, keepdims=True)

C1 = (X1 @ X1.T) / N1 - (m1 @ m1.T)
C2 = (X2 @ X2.T) / N2 - (m2 @ m2.T)

# Guard against singular matrices for random cases.
reg = 1e-10
C1i = np.linalg.inv(C1 + reg * np.eye(3))
C2i = np.linalg.inv(C2 + reg * np.eye(3))

# -----------------------------------------------------------------------------
# Distances / discriminants
# d1 = log(1/2) + X' inv(C1) m1 - 1/2 m1' inv(C1) m1
# d2 = log(1/2) + X' inv(C2) m2 - 1/2 m2' inv(C2) m2
# d12 = d1 - d2
# -----------------------------------------------------------------------------
w1 = C1i @ m1
b1 = np.log(0.5) - 0.5 * float((m1.T @ C1i @ m1).item())

w2 = C2i @ m2
b2 = np.log(0.5) - 0.5 * float((m2.T @ C2i @ m2).item())

w = w1 - w2
b = b1 - b2

print("d1(X) =", f"{w1[0,0]:.3f}*x1 + {w1[1,0]:.3f}*x2 + {w1[2,0]:.3f}*x3 + {b1:.3f}")
print("d2(X) =", f"{w2[0,0]:.3f}*x1 + {w2[1,0]:.3f}*x2 + {w2[2,0]:.3f}*x3 + {b2:.3f}")
print("d12(X)=", f"{w[0,0]:.3f}*x1 + {w[1,0]:.3f}*x2 + {w[2,0]:.3f}*x3 + {b:.3f}")

# Display
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Points and means
ax.scatter(X1[0, :], X1[1, :], X1[2, :], marker="o", s=90, edgecolors="k", facecolors="r")
ax.scatter(m1[0, 0], m1[1, 0], m1[2, 0], marker="s", s=90, edgecolors="k", facecolors="r")

ax.scatter(X2[0, :], X2[1, :], X2[2, :], marker="o", s=90, edgecolors="k", facecolors="g")
ax.scatter(m2[0, 0], m2[1, 0], m2[2, 0], marker="s", s=90, edgecolors="k", facecolors="g")

# Decision plane: w1*x + w2*y + w3*z + b = 0 -> z = -(w1*x + w2*y + b)/w3
les_x, les_y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
if abs(float(w[2, 0])) > 1e-12:
    les_z = -(float(w[0, 0]) * les_x + float(w[1, 0]) * les_y + b) / float(w[2, 0])
    color = 0.5 * np.ones((3, 3, 3), dtype=float)
    ax.plot_surface(les_x, les_y, les_z, facecolors=color, shade=False, alpha=0.8)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("X_1 (or), X_2 (og), μ_1 (sr), μ_2 (sg), X / d(X, X_1) == d(X, X_2) (k)")
ax.grid(True)

# MATLAB axis comment is optional; keep auto for choix=2 random case.

out_path = os.path.join(os.path.dirname(__file__), "Figure1320.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()

# -----------------------------------------------------------------------------
# LaTeX-style outputs (console)
# -----------------------------------------------------------------------------
def to_latex_matrix(A: np.ndarray) -> str:
    rows = []
    for r in A:
        rows.append(" & ".join(f"{float(v):.3f}" for v in np.ravel(r)))
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


print("latex(X1):", to_latex_matrix(X1))
print("latex(X2):", to_latex_matrix(X2))
print("latex(m1):", to_latex_matrix(m1))
print("latex(m2):", to_latex_matrix(m2))
print("latex(C1):", to_latex_matrix(C1))
print("latex(C2):", to_latex_matrix(C2))
print("latex(d1):", f"{w1[0,0]:.3f} x_1 + {w1[1,0]:.3f} x_2 + {w1[2,0]:.3f} x_3 + {b1:.3f}")
print("latex(d2):", f"{w2[0,0]:.3f} x_1 + {w2[1,0]:.3f} x_2 + {w2[2,0]:.3f} x_3 + {b2:.3f}")
print("latex(d12):", f"{w[0,0]:.3f} x_1 + {w[1,0]:.3f} x_2 + {w[2,0]:.3f} x_3 + {b:.3f}")
