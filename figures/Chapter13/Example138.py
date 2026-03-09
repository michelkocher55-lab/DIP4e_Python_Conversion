# Example 13.8

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

# Add current directory to path
sys.path.append(".")

from libDIP.perceptronClassifier4e import perceptronClassifier4e
from libDIP.perceptronTraining4e import perceptronTraining4e
from libDIPUM.data_path import dip_data

print("Running Example138...")

# 1. Load Data
data = scipy.io.loadmat(dip_data("fisheriris.mat"))
meas = data["meas"]
species = data["species"]

# Species might be arrays of arrays
species = np.array([s[0] for s in species.flatten()])

# 2. Select Classes
idx_setosa = species == "setosa"
idx_versicolor = species == "versicolor"

# 3. Form Input Matrix X
setosa_data = meas[idx_setosa, :2]
versicolor_data = meas[idx_versicolor, :2]

# 50 samples each?
X = np.concatenate((setosa_data, versicolor_data), axis=0).T  # (2, 100)

# Augment
X = np.vstack((X, np.ones((1, X.shape[1]))))  # (3, 100)

# Targets R
n_setosa = np.sum(idx_setosa)
n_versicolor = np.sum(idx_versicolor)

R = np.concatenate((np.ones(n_setosa), -np.ones(n_versicolor)))  # (100,)

# 4. Training
Alpha = 0.5
NEpochs = 1000
np.random.seed(0)
W0 = np.random.rand(3)

W, ActualNEpochs = perceptronTraining4e(X, R, Alpha, NEpochs, W0)

print(f"Converged in {ActualNEpochs} epochs.")
print(f"Weights: {W}")

# 6. Test
rout, numError, recogRate = perceptronClassifier4e(X, W, R)
print(f"Recog Rate: {recogRate}%")

# 7. Display
plt.figure("Perceptron 2D Iris Classification")
X_setosa = X[:2, :n_setosa]
X_versicolor = X[:2, n_setosa:]

plt.plot(X_versicolor[0], X_versicolor[1], "or", label="Versicolor")
plt.plot(X_setosa[0], X_setosa[1], "og", label="Setosa")

plt.axis("equal")
plt.title(f"Perceptron 2D Iris, {ActualNEpochs} epochs, Rate={recogRate}%")

# Plot Boundary
min_x = np.min(X[0, :])
max_x = np.max(X[0, :])

xs = np.linspace(min_x, max_x, 100)
ys = -(W[0] * xs + W[2]) / W[1]

plt.plot(xs, ys, "-k", linewidth=2, label="Boundary")
plt.legend()

plt.savefig("Example138.png")
print("Example138 Completed. Figure saved.")
plt.show()
