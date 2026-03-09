import numpy as np
import matplotlib.pyplot as plt
import sys

# Add current directory to path
sys.path.append(".")

from lib.neuralNet4e import neuralNet4e
from scipy import ndimage


def main():
    """main."""
    print("Running Example1316NN...")

    # 1. Generate Data (Same as Example1316CNN)
    # 3 Classes, 9 images.

    raw_data = np.array(
        [
            # Class 1
            [255, 0, 255, 255, 0, 255, 255, 0, 255],
            [198, 5, 213, 244, 14, 245, 241, 8, 231],
            [246, 8, 222, 225, 40, 237, 228, 5, 235],
            # Class 2
            [255, 255, 255, 255, 0, 255, 255, 255, 255],
            [234, 255, 205, 251, 0, 251, 238, 253, 240],
            [232, 255, 231, 247, 38, 246, 190, 236, 250],
            # Class 3
            [255, 255, 255, 0, 0, 0, 255, 255, 255],
            [245, 225, 205, 1, 0, 5, 238, 253, 240],
            [225, 235, 231, 7, 8, 4, 190, 236, 250],
        ]
    )

    # MATLAB: images = [...]'
    images = raw_data.T  # (9, 9)

    # Labels
    # MATLAB: 3x9
    labels = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
        ]
    )

    # Format Data
    # train_x = reshape(images, 3, 3, []) / 255.0
    # MATLAB reshape is F-order.
    # Python equivalent:
    train_x = images.reshape(3, 3, -1, order="F") / 255.0

    # Resize to 6x6
    train_x = ndimage.zoom(train_x, (2, 2, 1), order=1)  # (6, 6, 9)
    train_y = labels

    # Test Data
    test_x = images.reshape(3, 3, -1, order="F") / 255.0
    test_x = ndimage.zoom(test_x, (2, 2, 1), order=1)

    # Add Gaussian Noise
    sigma = np.sqrt(0.01)
    noise = np.random.normal(0, sigma, test_x.shape)
    test_x = test_x + noise
    test_x = np.clip(test_x, 0.0, 1.0)

    test_y = labels

    # Display Training Images
    plt.figure("Training Images", figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(train_x[:, :, i], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Train {i + 1}")
    plt.tight_layout()
    plt.savefig("Example1316NN_TrainImages.png")

    # Display Testing Images
    plt.figure("Testing Images", figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_x[:, :, i], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Test {i + 1}")
    plt.tight_layout()
    plt.savefig("Example1316NN_TestImages.png")

    # Prepare for Neural Net
    # Reshape to (N_features, N_samples)
    # Features = 6*6 = 36.
    # Flatten images.
    # MATLAB: reshape(train_x, [], 9)
    # F-order flatten if we want to match MATLAB exactly?
    # Python flattened images usually C-order.
    # `reshape(-1, 9)` in python will flatten whole array elements...
    # We want columns to be samples.
    # train_x is (6, 6, 9).
    # reshape(..., 9) implies we collapse 6,6 into 36.
    # F-order: fills column first (dim 0).
    X_train = train_x.reshape(-1, 9, order="F")  # (36, 9)
    X_test = test_x.reshape(-1, 9, order="F")  # (36, 9)

    # Specs
    # Layers=3. Nodes=[36, 32, 3].
    specs = {
        "Nodes": [36, 32, 3],
        "Activation": "sigmoid",
        "Correction": 1.0,
        "Mode": "train",
    }

    input_train = {"X": X_train, "R": train_y, "Epochs": 500}

    # Train
    print("Training NN (500 epochs)...")
    np.random.seed(0)
    output = neuralNet4e(input_train, specs)

    # Plot MSE
    plt.figure("MSE")
    plt.subplot(2, 1, 1)
    plt.plot(output["MSE"])
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training MSE")

    # Plot Results (Target vs Output) on subplot 2
    # output.Class vs vec2ind(test_y) ? No, test_y is target.
    # Wait, in the MATLAB code line 103, it plots Test Targets vs Train Output Class?
    # Or Output Class is from training phase?
    # "bar (1 : 9, [vec2ind(test_y); output.Class])"
    # Yes, it plots Targets (test_y) against Class (output.Class).
    # But output contains Class from the training pass (predictions on training data).

    # Targets (convert one-hot to index 0-2)
    targets = np.argmax(train_y, axis=0)  # 0-based
    preds = output["Class"]  # 0-based

    # Plot bars.
    # We can plot side-by-side bars.
    plt.subplot(2, 1, 2)
    x = np.arange(9)
    width = 0.35
    plt.bar(x - width / 2, targets, width, label="Target")
    plt.bar(x + width / 2, preds, width, label="Output")
    plt.title("Target vs Output (Training)")
    plt.legend()
    plt.savefig("Example1316NN_TrainingResults.png")

    # Classify Test Set
    print("Classifying Test Set...")
    specs["Mode"] = "classify"
    specs["W"] = output["W"]
    specs["b"] = output["b"]

    input_test = {
        "X": X_test,
        "Epochs": 1,  # Ignored
    }

    output_test = neuralNet4e(input_test, specs)

    # Print Results
    print("Test Classification Results:")
    print(output_test["Class"])

    # Calculate Test Accuracy
    test_targets = np.argmax(test_y, axis=0)
    test_preds = output_test["Class"]
    accuracy = np.mean(test_targets == test_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Show activation of last layer
    # output.A{3} -> last layer activations
    # In Python output['A'][L]. L=3 in MATLAB? Python indices?
    # neuralNet4e.py uses 1-based indexing for A/W/b for consistency logic?
    # Yes (list of size L+1).
    # Specs Nodes len = 3 (36, 32, 3). So L=3.
    # Activations in A[3].
    print("Output Activations (Test):")
    print(output_test["A"][3])

    print("Example1316NN Completed. Figures saved.")
    plt.show()


if __name__ == "__main__":
    main()
