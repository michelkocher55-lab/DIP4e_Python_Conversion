from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

# Add current directory to path
sys.path.append(".")

from libDIPUM.cnnsetup import cnnsetup
from libDIPUM.cnntrain import cnntrain
from libDIPUM.cnntest import cnntest
from libDIPUM.data_path import dip_data


def load_mnist_data(path: Any = None):
    """load_mnist_data."""
    path = dip_data("mnist_uint8.mat")
    print(f"Loading data from {path}...")
    data = scipy.io.loadmat(path)

    # Extract
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    # Preprocess
    train_x = train_x.astype(float) / 255.0
    test_x = test_x.astype(float) / 255.0

    # Reshape
    train_x = train_x.reshape(-1, 28, 28).transpose(1, 2, 0)
    test_x = test_x.reshape(-1, 28, 28).transpose(1, 2, 0)

    # Labels
    train_y = train_y.T.astype(float)
    test_y = test_y.T.astype(float)

    return train_x, train_y, test_x, test_y


print("Running Example1315CNN (MNIST)...")
# 1. Load Data
train_x, train_y, test_x, test_y = load_mnist_data()

print(f"Train Data: {train_x.shape}, Labels: {train_y.shape}")
print(f"Test Data: {test_x.shape}, Labels: {test_y.shape}")

# 2. Setup CNN
# Input 28x28
# Conv 5x5, 6 maps
# Sub 2
# Conv 5x5, 12 maps
# Sub 2
# FC -> 10

np.random.seed(0)

cnn = {}
cnn["layers"] = [
    {"type": "i"},  # 28x28
    {"type": "c", "outputmaps": 6, "kernelsize": 5},  # -> 24x24
    {"type": "s", "scale": 2},  # -> 12x12
    {"type": "c", "outputmaps": 12, "kernelsize": 5},  # -> 8x8
    {"type": "s", "scale": 2},  # -> 4x4
]
# FC input dim: 4*4*12 = 192.

cnn = cnnsetup(cnn, train_x, train_y)

# 3. Train
# MATLAB: 1 Epoch. Batch size 50. Alpha 1.
opts = {"alpha": 1.0, "batchsize": 50, "numepochs": 1}

print("Training (1 epoch)...")
cnn = cnntrain(cnn, train_x, train_y, opts)

# 4. Test
err_train, cnn, bad_train, est_train = cnntest(cnn, train_x, train_y)
print(f"Error rate (training) = {err_train * 100:.2f}%")

err_test, cnn, bad_test, est_test = cnntest(cnn, test_x, test_y)
print(f"Error rate (testing) = {err_test * 100:.2f}%")

# 5. Plots

# Batch-wise MSE
plt.figure("Training MSE")
plt.plot(cnn["rL"], linewidth=2)
plt.xlabel("Batches")
plt.ylabel("Mean Squared Error")
plt.title("Batch-wise mean squared error during training")
plt.grid(True)
plt.savefig("Example1315CNN_MSE.png")

# Class-wise Accuracy (Test)
truth_labels = np.argmax(test_y, axis=0)
h1 = np.zeros(10)  # Total
h2 = np.zeros(10)  # Correct

for i in range(len(truth_labels)):
    cls = truth_labels[i]
    h1[cls] += 1
    if cls == est_test[i]:
        h2[cls] += 1

plt.figure("Class Accuracy (Test)")
x = np.arange(10)
width = 0.35
plt.bar(x - width / 2, h1, width, label="Total Samples")
plt.bar(x + width / 2, h2, width, label="Correct Samples")
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class-wise Classification Accuracy (Test)")
plt.xticks(x)
plt.legend()

# Add text
accuracy = np.divide(h2, h1, out=np.zeros_like(h1), where=h1 != 0)
for i, acc in enumerate(accuracy):
    plt.text(x[i], h1[i] + 50, f"{acc:.2f}", ha="center", fontsize=8)

plt.savefig("Example1315CNN_Accuracy.png")

# Visualize Kernels (Layer 1)
layer1 = cnn["layers"][1]
if layer1["type"] == "c":
    kernels = layer1["k"]
    num_input = len(kernels)  # 1
    num_output = len(kernels[0])  # 6

    plt.figure("Kernels Layer 1")
    for k in range(num_output):
        plt.subplot(1, num_output, k + 1)
        plt.imshow(kernels[0][k], cmap="gray")
        plt.axis("off")
        plt.title(f"K{k + 1}")
    plt.suptitle("Layer 1 Kernels")
    plt.savefig("Example1315CNN_Kernels_L1.png")

# Visualize Kernels (Layer 2): 6 input maps x 12 output maps
layer2 = cnn["layers"][3]
if layer2["type"] == "c":
    kernels2 = layer2["k"]
    num_input2 = len(kernels2)  # expected 6
    num_output2 = len(kernels2[0])  # expected 12

    fig_l2 = plt.figure("Kernels Layer 2", figsize=(18, 10))
    for i in range(num_input2):
        for j in range(num_output2):
            ax = plt.subplot(num_input2, num_output2, i * num_output2 + j + 1)
            ax.imshow(kernels2[i][j], cmap="gray")
            ax.axis("off")
    fig_l2.suptitle("Layer 2 Kernels (rows=input maps, cols=output maps)")
    fig_l2.tight_layout(rect=[0, 0, 1, 0.96])
    fig_l2.savefig("Example1315CNN_Kernels_L2.png")

print("Example1315CNN Completed. Figures saved.")

# Show
plt.show()
