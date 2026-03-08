"""Figure 13.26 - LMSE perceptron error plots for separable and inseparable Iris data."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from libDIP.lmsePerceptronTraining4e import lmsePerceptronTraining4e
from libDIP.perceptronClassifier4e import perceptronClassifier4e
from libDIPUM.data_path import dip_data

print("Running Figure1326 (LMSE perceptron errors on iris data)...")

MAT_PATH = dip_data('fisheriris.mat')


# Parameters
inputSep: dict[str, object] = {
    "W0": np.zeros((5, 1), dtype=float),
    "Nepochs": 900,
    "Alpha": 0.001,
}
inputNonSep: dict[str, object] = {
    "W0": np.zeros((5, 1), dtype=float),
    "Nepochs": 900,
    "Alpha": 0.001,
}

# Data
data = loadmat(MAT_PATH)
meas = np.asarray(data["meas"], dtype=np.float64)
meas = meas.T  # Patterns as columns: 4 x 150

# Separable: setosa vs versicolor (first 100 samples)
inputSep["X"] = meas[:, 0:100].copy()

# Nonseparable: versicolor vs virginica (samples 51:150)
inputNonSep["X"] = meas[:, 50:150].copy()

# Class vectors
R_sep = np.empty(100, dtype=float)
R_sep[0:50] = 1
R_sep[50:100] = -1
inputSep["R"] = R_sep

R_nonsep = np.empty(100, dtype=float)
R_nonsep[0:50] = 1
R_nonsep[50:100] = -1
inputNonSep["R"] = R_nonsep

# Augment vectors by 1
X_sep = np.asarray(inputSep["X"], dtype=float)
X_nonsep = np.asarray(inputNonSep["X"], dtype=float)
X_sep_aug = np.vstack((X_sep, np.ones((1, X_sep.shape[1]), dtype=float)))
X_nonsep_aug = np.vstack((X_nonsep, np.ones((1, X_nonsep.shape[1]), dtype=float)))
inputSep["X"] = X_sep_aug
inputNonSep["X"] = X_nonsep_aug

# Training
outputSep = lmsePerceptronTraining4e(inputSep)
outputNonSep = lmsePerceptronTraining4e(inputNonSep)

# Training-set recognition
routSep, numErrorsSep, recogRateSep = perceptronClassifier4e(
    np.asarray(inputSep["X"]), outputSep["W"], np.asarray(inputSep["R"])
)
routNonSep, numErrorsNonSep, recogRateNonSep = perceptronClassifier4e(
    np.asarray(inputNonSep["X"]), outputNonSep["W"], np.asarray(inputNonSep["R"])
)

print(f"Separable set (setosa vs versicolor): errors = {int(numErrorsSep)}, recognition = {recogRateSep:.2f}%")
print(f"Nonseparable set (versicolor vs virginica): errors = {int(numErrorsNonSep)}, recognition = {recogRateNonSep:.2f}%")

# Display
fig = plt.figure(1, figsize=(10, 4.5))

ax1 = fig.add_subplot(1, 2, 1)
err_sep = np.asarray(outputSep["Error"]).reshape(-1)
ax1.plot(np.arange(1, err_sep.size + 1), err_sep, "k-")
ax1.set_title("Line Separable")
ax1.set_xlim([1, 50])
ax1.set_ylim([0, 0.3])

ax2 = fig.add_subplot(1, 2, 2)
err_nonsep = np.asarray(outputNonSep["Error"]).reshape(-1)
ax2.plot(np.arange(1, err_nonsep.size + 1), err_nonsep, "k-")
ax2.set_title("Line non separable")
ax2.set_xlim([1, 900])
ax2.set_ylim([0, 0.3])

out_path = os.path.join(os.path.dirname(__file__), "Figure1326.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
