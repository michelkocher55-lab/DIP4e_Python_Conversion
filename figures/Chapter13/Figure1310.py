"""Figure 13.10 - Minimum distance classifier (Iris setosa vs versicolor)."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from libDIPUM.data_path import dip_data


print("Running Figure1310 (minimum distance classifier)...")

# Data
mat_path = dip_data('fisheriris.mat')


def _load_fisheriris(path: str):
    try:
        data = loadmat(path, simplify_cells=True)
    except TypeError:
        # Older SciPy without simplify_cells
        data = loadmat(path, squeeze_me=True, struct_as_record=False)

    if "meas" not in data or "species" not in data:
        raise KeyError("fisheriris.mat must contain 'meas' and 'species'.")

    meas = np.asarray(data["meas"], dtype=np.float64)
    species_raw = data["species"]

    if isinstance(species_raw, np.ndarray):
        species = [str(s) for s in species_raw.ravel().tolist()]
    else:
        species = [str(s) for s in list(species_raw)]

    species = [s.strip() for s in species]
    return meas, np.asarray(species, dtype=object)


meas, species = _load_fisheriris(mat_path)

# Keep only setosa and versicolor, using petal length/width (cols 3:4 in MATLAB).
ix1 = species == "setosa"
ix2 = species == "versicolor"
X1 = meas[ix1, 2:4]  # Class 1 = setosa
X2 = meas[ix2, 2:4]  # Class 2 = versicolor

# Mean and decision function d12(x1,x2) = a*x1 + b*x2 + c
m1 = np.mean(X1, axis=0).reshape(2, 1)
m2 = np.mean(X2, axis=0).reshape(2, 1)

a = float(m1[0, 0] - m2[0, 0])
b = float(m1[1, 0] - m2[1, 0])
c = -0.5 * float((m1.T @ m1 - m2.T @ m2).item())

print(f"d12(x1, x2) ≈ {a:.3f}*x1 + {b:.3f}*x2 + {c:.3f}")

# Classification of one random test sample
min_petal_length = float(np.min(np.r_[X1[:, 0], X2[:, 0]]))
max_petal_length = float(np.max(np.r_[X1[:, 0], X2[:, 0]]))
min_petal_width = float(np.min(np.r_[X1[:, 1], X2[:, 1]]))
max_petal_width = float(np.max(np.r_[X1[:, 1], X2[:, 1]]))

xt = min_petal_length + np.random.rand() * (max_petal_length - min_petal_length)
yt = min_petal_width + np.random.rand() * (max_petal_width - min_petal_width)
X_test = np.array([[xt, yt]], dtype=np.float64)

d1_test = X_test @ m1 - 0.5 * (m1.T @ m1)
d2_test = X_test @ m2 - 0.5 * (m2.T @ m2)
d12_test = float((d1_test - d2_test).item())

if d12_test > 0:
    test_class = "setosa"
    test_marker = "sr"  # square red
else:
    test_class = "versicolor"
    test_marker = "sg"  # square green

# Display (MATLAB ezplot equivalent for d12=0)
fig = plt.figure(1)

if abs(b) > 1e-12:
    xx = np.linspace(min_petal_length, max_petal_length, 400)
    yy = -(a * xx + c) / b
    plt.plot(xx, yy, "b-")
else:
    x_const = -c / a if abs(a) > 1e-12 else min_petal_length
    plt.plot([x_const, x_const], [min_petal_width, max_petal_width], "b-")

plt.plot(X1[:, 0], X1[:, 1], "or", X2[:, 0], X2[:, 1], "*g", X_test[0, 0], X_test[0, 1], test_marker)
plt.xlabel("Petal length [cm]")
plt.ylabel("Petal width [cm]")

m1_txt = np.array2string(m1.ravel(), precision=2, suppress_small=True)
m2_txt = np.array2string(m2.ravel(), precision=2, suppress_small=True)
d12_txt = f"{a:.3f}*x1 + {b:.3f}*x2 + {c:.3f}"

plt.title(
    f"Iris_s (or), μ = {m1_txt}, Iris_v (*g), μ = {m2_txt} "
    f"Test (sb) in {test_class}, {d12_txt} = 0"
)
plt.xlim([min_petal_length, max_petal_length])
plt.ylim([min_petal_width, max_petal_width])

out_path = os.path.join(os.path.dirname(__file__), "Figure1310.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()

# LaTeX-like output strings (no sympy dependency)
m1_latex = r"\begin{bmatrix}" + f"{m1[0,0]:.3f}\\\\{m1[1,0]:.3f}" + r"\end{bmatrix}"
m2_latex = r"\begin{bmatrix}" + f"{m2[0,0]:.3f}\\\\{m2[1,0]:.3f}" + r"\end{bmatrix}"
d12_latex = f"{a:.3f} x_1 + {b:.3f} x_2 + {c:.3f}"
print("latex(m1):", m1_latex)
print("latex(m2):", m2_latex)
print("latex(d12):", d12_latex)
