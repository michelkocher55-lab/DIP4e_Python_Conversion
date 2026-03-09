"""Figure 13.34 - XOR gate using neuralNet4e and DIP4e helpers."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from libDIP.neuralNet4e import neuralNet4e
from libDIP.moreTrainingPatterns4e import moreTrainingPatterns4e
from libDIP.patternShuffle4e import patternShuffle4e


print("Running Figure1334 (XOR with neuralNet4e)...")

# Parameters
HiddenSizes = 2
NRep = 100
NEpochs = 100
Correction = 0.02
NDisp = 40

# Data
Input = np.array([[1, -1, -1, 1], [1, -1, 1, -1]], dtype=float)
Target = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)

# NN training (DIP4e)
Specs = {
    "Nodes": [2, 2, 2],
    "Activation": "sigmoid",
    "Mode": "train",
    # MATLAB-like indexing used by neuralNet4e: W[2], W[3], b[2], b[3]
    "W": [None, None, np.random.rand(HiddenSizes, 2), np.random.rand(2, HiddenSizes)],
    "b": [None, None, np.random.rand(HiddenSizes, 1), np.random.rand(2, 1)],
    "Correction": Correction,
}

Inputt = {"X": Input, "R": Target}
Inputt["X"], Inputt["R"] = moreTrainingPatterns4e(Inputt["X"], Inputt["R"], NRep)
Inputt["Epochs"] = NEpochs

MSE = []
for _ in range(5):
    Output = neuralNet4e(Inputt, Specs)
    Inputt["X"], Inputt["R"], order = patternShuffle4e(
        Inputt["X"], Inputt["R"], "random"
    )
    Specs["W"] = Output["W"]
    Specs["b"] = Output["b"]
    MSE.extend(Output.get("MSE", []))

print(f"First layer weights = {Specs['W'][2]}")
print(f"Second layer weights = {Specs['W'][3]}")
print(f"First layer biases = {Specs['b'][2]}")
print(f"Second layer biases = {Specs['b'][3]}")

# NN test (DIP4e): output for 4 inputs
SpecsTest = {
    "Nodes": Specs["Nodes"],
    "Mode": "test",
    "Activation": "sigmoid",
    "W": Output["W"],
    "b": Output["b"],
}
InputTest = {"X": Input, "R": Target}
OutputTest = neuralNet4e(InputTest, SpecsTest)

print(f"Input = {Input}")
print(f"Output = {OutputTest['A'][3][0, :]}")
print(f"Recognition rate = {OutputTest['RecogRate']}")

# NN test (DIP4e): output on meshgrid
LesX = np.linspace(-1, 1, NDisp)
LesY = np.linspace(-1, 1, NDisp)
X, Y = np.meshgrid(LesX, LesY)

# MATLAB-equivalent vectorization uses column-major order.
MyInputTest = {
    "X": np.vstack((X.reshape(1, -1, order="F"), Y.reshape(1, -1, order="F")))
}
MyOutputTest = neuralNet4e(MyInputTest, SpecsTest)
Z = np.reshape(MyOutputTest["A"][3][0, :], (NDisp, NDisp), order="F")

# Display
fig = plt.figure(1, figsize=(13, 4.4))

ax1 = fig.add_subplot(1, 3, 1)
ax1.plot([-1, 1], [-1, 1], "or")
ax1.plot([-1, 1], [1, -1], "og")
ax1.set_box_aspect(1)
ax1.set_title("XOR Gate")

ax2 = fig.add_subplot(1, 3, 2)
ax2.contour(X, Y, Z)
ax2.set_xlabel("x_1")
ax2.set_ylabel("x_2")
ax2.set_box_aspect(1)
ax2.set_title("XOR Gate")

ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax3.set_xlabel("x_1")
ax3.set_ylabel("x_2")
ax3.set_zlabel("output")
ax3.set_title("XOR Gate")
ax3.set_box_aspect((1, 1, 1))

out_path = os.path.join(os.path.dirname(__file__), "Figure1334.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
