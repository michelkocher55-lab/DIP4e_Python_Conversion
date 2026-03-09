from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys
import os

# Add current directory to path
sys.path.append(".")

from lib.neuralNet4e import neuralNet4e
from libDIPUM.data_path import dip_data


def load_mnist_data(path: Any = None):
    """Load and preprocess MNIST data from .mat file for NN."""
    if path is None:
        path = dip_data("mnist_uint8.mat")

    if not os.path.exists(path):
        print(f"Error: MNIST file not found at {path}")
        sys.exit(1)

    print(f"Loading data from {path}...")
    data = scipy.io.loadmat(path)

    # Extract
    train_x = data["train_x"]  # (60000, 784)
    train_y = data["train_y"]  # (60000, 10)
    test_x = data["test_x"]
    test_y = data["test_y"]

    # NN Expects:
    # Input X: (N_features, N_samples)
    # Target R: (N_classes, N_samples)

    # Preprocess
    # Scale to 0-1
    train_x = train_x.astype(float) / 255.0
    test_x = test_x.astype(float) / 255.0

    # Transpose to (Features, Samples)
    train_x = train_x.T
    test_x = test_x.T

    train_y = train_y.T.astype(float)  # (10, 60000)
    test_y = test_y.T.astype(float)

    return train_x, train_y, test_x, test_y


def main():
    """main."""
    print("Running Example1315NN (MNIST Neural Net)...")

    # 1. Load Data
    train_x, train_y, test_x, test_y = load_mnist_data()

    print(f"Train Data: {train_x.shape}, Labels: {train_y.shape}")
    print(f"Test Data: {test_x.shape}, Labels: {test_y.shape}")

    # 2. Parameters
    # specs.Layers = 4? specs.Nodes = [28^2, 1024, 512, 10]
    specs = {
        "Nodes": [28 * 28, 1024, 512, 10],
        "Activation": "sigmoid",
        "Correction": 0.1,
        "Mode": "train",
    }

    input_data = {
        "X": train_x,  # (784, 60000)
        "R": train_y,  # (10, 60000)
        "Epochs": 10,  # Example uses 10
    }

    # 3. Train
    print("Training NN (10 epochs)...")
    # This might be slow in Python for large NN!
    # 784->1024->512->10 fully connected.
    # W1: 1024x784. W2: 512x1024. W3: 10x512.
    # Batch size: neuralNet4e uses ALL patterns in one batch!
    # "compute the error LMSE error for all patterns"
    # It does full batch gradient descent on 60,000 images!
    # Matrix multiplications: (1024, 784) * (784, 60000).
    # This is heavy.
    # MATLAB handles large matrix mul somewhat efficiently.
    # Numpy should be okay too (BLAS).

    output = neuralNet4e(input_data, specs)
    MSE = output["MSE"]

    # 4. Classify Test Set
    print("Classifying Test Set...")
    specs["Mode"] = "classify"
    specs["W"] = output["W"]
    specs["b"] = output["b"]

    input_test = {
        "X": test_x,
        "Epochs": 1,  # Ignored
    }

    output_test = neuralNet4e(input_test, specs)

    # Calculate Accuracy manually
    truth = np.argmax(test_y, axis=0)  # 0-9
    preds = output_test["Class"]

    accuracy = np.mean(truth == preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 5. Display
    # Montage of first 50 test images
    # test_x is (784, 10000).
    imgs = test_x[:, :50]
    # Reshape to (28, 28, 50).
    # Need to match reshape logic.
    # In loading, we just transposed (N, 784) -> (784, N).
    # So each column is a flattened image (row-major original).
    imgs = imgs.T.reshape(50, 28, 28)

    plt.figure("Test Images (First 50)", figsize=(10, 5))
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("Example1315NN_Montage.png")

    # MSE Plot
    plt.figure("MSE")
    plt.plot(MSE)
    plt.xlabel("Epoch")
    plt.title("MSE")
    plt.grid(True)
    plt.savefig("Example1315NN_MSE.png")

    print("Example1315NN Completed. Figures saved.")
    plt.show()


if __name__ == "__main__":
    main()
