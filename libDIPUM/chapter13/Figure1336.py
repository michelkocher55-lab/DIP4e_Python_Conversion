"""Figure 13.36 - Error plot for XOR problem."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from libDIP.neuralNet4e import neuralNet4e


print("Running Figure1336 (XOR MSE plot)...")

# Parameters
input_data = {"Epochs": 1000}
specs = {
    "Layers": 3,
    "Nodes": [2, 2, 2],
    "Correction": 1.0,
}

# Data (columns are patterns)
input_data["X"] = np.array([[-1, 1, -1, 1], [-1, 1, 1, -1]], dtype=float)
input_data["R"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)

# Train network
specs["Mode"] = "train"
output = neuralNet4e(input_data, specs)

# Display
fig = plt.figure(1)
plt.plot(np.asarray(output["MSE"]).reshape(-1), "k-")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("The XOR Problem")

# Save
out_path = os.path.join(os.path.dirname(__file__), "Figure1336.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
