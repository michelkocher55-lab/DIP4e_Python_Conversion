from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter13Mixin:
    def example1315cnn(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Example1315CNN.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            import scipy.io
            import sys

            # Add current directory to path
            sys.path.append(".")

            from helpers.cnnsetup import cnnsetup
            from helpers.cnntrain import cnntrain
            from helpers.cnntest import cnntest
            from helpers.data_path import dip_data

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
                        ax = plt.subplot(
                            num_input2, num_output2, i * num_output2 + j + 1
                        )
                        ax.imshow(kernels2[i][j], cmap="gray")
                        ax.axis("off")
                fig_l2.suptitle("Layer 2 Kernels (rows=input maps, cols=output maps)")
                fig_l2.tight_layout(rect=[0, 0, 1, 0.96])
                fig_l2.savefig("Example1315CNN_Kernels_L2.png")

            print("Example1315CNN Completed. Figures saved.")

            # Show
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def example1315nn(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Example1315NN.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            import scipy.io
            import sys
            import os

            # Add current directory to path
            sys.path.append(".")

            from lib.neuralNet4e import neuralNet4e
            from helpers.data_path import dip_data

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
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def example1316cnn(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Example1316CNN.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            from helpers.cnnsetup import cnnsetup
            from helpers.cnntrain import cnntrain
            from helpers.cnntest import cnntest

            print("Running Example1316CNN...")

            # 1. Generate Training and Testing Images
            num_classes = 3
            num_images_per_class = 3

            # Raw Data Definition
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

            images = raw_data.T  # (9, 9)

            # Labels
            labels = np.array(
                [
                    [1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1],
                ]
            )

            # Format Data
            train_x = images.reshape(3, 3, -1, order="F") / 255.0

            # Resize to 6x6
            train_x = ndimage.zoom(train_x, (2, 2, 1), order=1)
            train_y = labels

            # Test Generation
            test_x = images.reshape(3, 3, -1, order="F") / 255.0
            test_x = ndimage.zoom(test_x, (2, 2, 1), order=1)

            # Add Gaussian Noise
            sigma = np.sqrt(0.01)
            noise = np.random.normal(0, sigma, test_x.shape)
            test_x = test_x + noise

            # Clip to 0-1
            test_x = np.clip(test_x, 0.0, 1.0)
            test_y = labels

            # Setup CNN
            # input (6x6) -> Conv (3x3, 2 kernels) -> 4x4x2
            # -> Sub (2x2) -> 2x2x2
            # -> Flatten -> 8
            # -> FC -> 3
            np.random.seed(0)

            cnn = {}
            cnn["layers"] = [
                {"type": "i"},
                {"type": "c", "outputmaps": 2, "kernelsize": 3},
                {"type": "s", "scale": 2},
            ]

            cnn = cnnsetup(cnn, train_x, train_y)

            # Train
            opts = {"alpha": 1.0, "batchsize": 9, "numepochs": 400}

            print("Training...")
            cnn = cnntrain(cnn, train_x, train_y, opts)

            # Test
            err_train, cnn, bad_train, est_train = cnntest(cnn, train_x, train_y)
            print(f"Error rate (training) = {err_train * 100:.2f}%")

            err_test, cnn, bad_test, est_test = cnntest(cnn, test_x, test_y)
            print(f"Error rate (testing) = {err_test * 100:.2f}%")

            # Class-wise Accuracy (Testing)
            # est_test: indices of predictions.
            # truth: indices of truth.
            truth_labels = np.argmax(test_y, axis=0)

            # Accuracy
            classes = np.unique(truth_labels)
            num_classes = len(classes)

            h1 = np.zeros(num_classes)  # Total
            h2 = np.zeros(num_classes)  # Correct

            for i in range(len(truth_labels)):
                cls = truth_labels[i]
                h1[cls] += 1
                if cls == est_test[i]:
                    h2[cls] += 1
            accuracy = np.divide(h2, h1, out=np.zeros_like(h1), where=h1 != 0)

            # Display
            plt.figure("Training Images", figsize=(6, 6))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(train_x[:, :, i], cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
                plt.title(f"Train / Class {i // 3 + 1}")
            plt.tight_layout()
            plt.savefig("Example1316CNN_Training_Images.png")

            plt.figure("Testing Images", figsize=(6, 6))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(test_x[:, :, i], cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
                plt.title(f"Test {i + 1}")
            plt.tight_layout()
            plt.savefig("Example1316CNN_Testing_Images.png")

            plt.figure("MSE")
            plt.plot(cnn["rL"], linewidth=2)
            plt.xlabel("Batches")
            plt.ylabel("Mean Squared Error")
            plt.title("Training MSE")
            plt.grid(True)
            plt.savefig("Example1316CNN_MSE.png")

            plt.figure("Class Accuracy")
            x = np.arange(num_classes) + 1
            width = 0.35
            plt.bar(x - width / 2, h1, width, label="Total Samples")
            plt.bar(x + width / 2, h2, width, label="Correct Samples")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title("Class-wise Accuracy")
            plt.xticks(x)
            plt.legend()

            for i, acc in enumerate(accuracy):
                plt.text(x[i], h1[i] + 0.1, f"{acc:.2f}", ha="center")

            plt.savefig("Example1316CNN_Accuracy.png")

            # Visualize Kernels
            # Layer 1 is 'c'
            layer1 = cnn["layers"][1]
            if layer1["type"] == "c":
                kernels = layer1["k"]  # list of lists
                num_input = len(kernels)
                num_output = len(kernels[0])

                plt.figure("Kernels")
                idx = 1
                for j in range(num_input):
                    for k in range(num_output):
                        plt.subplot(num_input, num_output, idx)
                        plt.imshow(kernels[j][k], cmap="gray")
                        plt.axis("off")
                        idx += 1
                plt.suptitle(f"Kernels: {num_input} in -> {num_output} out")
                plt.savefig("Example1316CNN_Kernels.png")

            # Feedforward Example
            # Plot maps for Sample 1
            sample_idx = 0
            fig_ff = plt.figure("FeedForward Example", figsize=(10, 6))

            # Input
            plt.subplot(2, 4, 1)
            plt.imshow(cnn["layers"][0]["a"][0][:, :, sample_idx], cmap="gray")
            plt.title("Input")
            plt.axis("off")

            # Conv Maps (2 maps)
            # 4x4
            for k in range(2):
                plt.subplot(2, 4, 2 + k)
                plt.imshow(cnn["layers"][1]["a"][k][:, :, sample_idx], cmap="gray")
                plt.title(f"Conv {k + 1}")
                plt.axis("off")

            # Sub Maps (2 maps)
            # 2x2
            for k in range(2):
                plt.subplot(2, 4, 5 + k)
                plt.imshow(cnn["layers"][2]["a"][k][:, :, sample_idx], cmap="gray")
                plt.title(f"Sub {k + 1}")
                plt.axis("off")

            # Output
            plt.subplot(2, 4, 8)
            out_probs = cnn["o"][:, sample_idx]
            plt.bar(range(1, 4), out_probs)
            plt.title("Output Probs")

            plt.tight_layout()
            plt.savefig("Example1316CNN_FeedForward.png")

            print("Example1316CNN Completed. Figures saved.")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def example1316nn(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Example1316NN.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
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
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def example138(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Example138.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            # Example 13.8

            import numpy as np
            import matplotlib.pyplot as plt
            import scipy.io
            import sys

            # Add current directory to path
            sys.path.append(".")

            from DIP4eFigures.perceptronClassifier4e import perceptronClassifier4e
            from DIP4eFigures.perceptronTraining4e import perceptronTraining4e
            from helpers.data_path import dip_data

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
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1308(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1308.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.im2minperpoly import im2minperpoly
            from helpers.randvertex import randvertex
            from helpers.polyangles import polyangles
            from helpers.strsimilarity import strsimilarity
            from helpers.data_path import dip_data

            # Parameters
            Nr = 10
            q = 45
            MaxDeviation = 9

            # Data
            path_f1 = dip_data("Fig1203(a)(bottle_1).tif")
            path_f2 = dip_data("Fig1203(d)(bottle_2).tif")

            f1 = imread(path_f1)
            f2 = imread(path_f2)

            # MPP
            print("Computing MPP for f1...")
            X1, Y1, R1 = im2minperpoly(f1, 8)
            print(f"MPP1 vertices: {len(X1)}")

            print("Computing MPP for f2...")
            X2, Y2, R2 = im2minperpoly(f2, 8)
            print(f"MPP2 vertices: {len(X2)}")

            # Noise Adding (Skipping full Nr loop for visual check first, or just do one)
            Xn1_list, Yn1_list = [], []
            Xn2_list, Yn2_list = [], []

            np.random.seed(0)  # rng default

            for r in range(Nr):
                xn1, yn1 = randvertex(X1, Y1, MaxDeviation)
                Xn1_list.append(xn1)
                Yn1_list.append(yn1)

                xn2, yn2 = randvertex(X2, Y2, MaxDeviation)
                Xn2_list.append(xn2)
                Yn2_list.append(yn2)

            # Signature Computation
            Angles1 = polyangles(X1, Y1)
            Angles2 = polyangles(X2, Y2)

            # Store noisy angles
            AnglesN1 = []
            AnglesN2 = []

            for r in range(Nr):
                temp1 = polyangles(Xn1_list[r], Yn1_list[r])
                AnglesN1.append(temp1)

                temp2 = polyangles(Xn2_list[r], Yn2_list[r])
                AnglesN2.append(temp2)

            # String conversion

            def vec2str(v: Any):
                """vec2str."""
                # Emulate int2str behavior for vector
                # "1  2  3"
                return "  ".join([str(int(x)) for x in v])

            s1_vec = np.floor(Angles1 / q) + 1
            s1 = vec2str(s1_vec)

            s2_vec = np.floor(Angles2 / q) + 1
            s2 = vec2str(s2_vec)

            sN1 = [vec2str(np.floor(a / q) + 1) for a in AnglesN1]
            sN2 = [vec2str(np.floor(a / q) + 1) for a in AnglesN2]

            # Similarity
            R12, _, _ = strsimilarity(s1, s2)
            print(f"Similarity s1-s2: {R12}")

            R1N1 = []
            for r in range(Nr):
                val, _, _ = strsimilarity(s1, sN1[r])
                R1N1.append(val)

            print(f"Mean Similarity s1-sN1: {np.mean(R1N1)}")

            # Display
            plt.figure(figsize=(15, 10))

            # Subplot 1: f1
            plt.subplot(2, 3, 1)
            plt.imshow(f1, cmap="gray")
            plt.title("f1")

            # Subplot 2: Bound f1
            plt.subplot(2, 3, 2)
            plt.plot(np.append(Y1, Y1[0]), np.append(X1, X1[0]), ".-")
            plt.gca().invert_yaxis()  # Match image coords
            plt.title(f"Bound f1, {len(X1)}")
            plt.axis("equal")

            # Subplot 3: Noisy f1
            plt.subplot(2, 3, 3)
            xn_disp = Xn1_list[0]
            yn_disp = Yn1_list[0]
            plt.plot(
                np.append(yn_disp, yn_disp[0]), np.append(xn_disp, xn_disp[0]), ".-"
            )
            plt.gca().invert_yaxis()
            plt.title(f"Noisy f1, {len(xn_disp)}")
            plt.axis("equal")

            # Subplot 4: f2
            plt.subplot(2, 3, 4)
            plt.imshow(f2, cmap="gray")
            plt.title("f2")

            # Subplot 5: Bound f2
            plt.subplot(2, 3, 5)
            plt.plot(np.append(Y2, Y2[0]), np.append(X2, X2[0]), ".-")
            plt.gca().invert_yaxis()
            plt.title(f"Bound f2, {len(X2)}")
            plt.axis("equal")

            # Subplot 6: Noisy f2
            plt.subplot(2, 3, 6)
            xn_disp2 = Xn2_list[0]
            yn_disp2 = Yn2_list[0]
            plt.plot(
                np.append(yn_disp2, yn_disp2[0]), np.append(xn_disp2, xn_disp2[0]), ".-"
            )
            plt.gca().invert_yaxis()
            plt.title(f"Noisy f2, {len(xn_disp2)}")
            plt.axis("equal")

            plt.tight_layout()
            plt.savefig("Figure1308.png")
            print("Figure saved to Figure1308.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1310(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1310.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.10 - Minimum distance classifier (Iris setosa vs versicolor)."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.io import loadmat
            from helpers.data_path import dip_data

            print("Running Figure1310 (minimum distance classifier)...")

            # Data
            mat_path = dip_data("fisheriris.mat")

            def _load_fisheriris(path: str):
                """_load_fisheriris."""
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

            xt = min_petal_length + np.random.rand() * (
                max_petal_length - min_petal_length
            )
            yt = min_petal_width + np.random.rand() * (
                max_petal_width - min_petal_width
            )
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

            plt.plot(
                X1[:, 0],
                X1[:, 1],
                "or",
                X2[:, 0],
                X2[:, 1],
                "*g",
                X_test[0, 0],
                X_test[0, 1],
                test_marker,
            )
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

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1310.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()

            # LaTeX-like output strings (no sympy dependency)
            m1_latex = (
                r"\begin{bmatrix}"
                + f"{m1[0, 0]:.3f}\\\\{m1[1, 0]:.3f}"
                + r"\end{bmatrix}"
            )
            m2_latex = (
                r"\begin{bmatrix}"
                + f"{m2[0, 0]:.3f}\\\\{m2[1, 0]:.3f}"
                + r"\end{bmatrix}"
            )
            d12_latex = f"{a:.3f} x_1 + {b:.3f} x_2 + {c:.3f}"
            print("latex(m1):", m1_latex)
            print("latex(m2):", m2_latex)
            print("latex(d12):", d12_latex)
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1311(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1311.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.signal import filtfilt

            # Init
            np.random.seed(0)  # rng('default')

            # Parameters
            Choice = 1
            SigmaNoise = 0.5

            Num = np.array([0.1, 1.0, 0.1])
            Den = np.array([1.0, 0.0, 0.0]) * np.sum(Num)

            # Signature Data (10x11 matrix)
            Signature = np.array(
                [
                    [0, 2, -2, -0.3, -0.2, 0, 0, 1.8, -2, 0, 0],
                    [0, 0, 1, 0, 1, -1, -2, 0, 0, 0, 0],
                    [0, 0, 2, -1, -0.5, 1, -2, 0, 0, 0, 0],
                    [0, 0, 1.3, 1, -2, -0.2, 0, -1, 0, 0, 0],
                    [0, 0, 1.2, 0, -1, 0, 2, 0, -2.5, 0, 0],
                    [0, 0, 1.5, -1, -0.2, -0.2, 1, -2, 0, 0, 0],
                    [0, 0, 1, -0.6, 0.3, -0.5, 0, 2, -3, 0, 0],
                    [0, 0, 1.2, -1, 1, -2, 0.5, -1, 0, 0, 0],
                    [0, 0, 1.2, 1.2, -2, -0.2, 0, 2, -2, -2, 0],
                    [0, 0, 2, -1, -1.8, 0, 0, 1, -2.5, 0, 0],
                ]
            )

            NSymbols, N = Signature.shape
            k = np.arange(1, N + 1)

            # Signatures
            if Choice:
                SignatureBlurred = filtfilt(Num, Den, Signature, axis=0)
            else:
                SignatureBlurred = Signature.copy()

            # Classification Stats (Monte Carlo)
            Nr = 100
            ConfusionMatrix = np.zeros((NSymbols, NSymbols), dtype=int)
            TotalErrors = 0

            # Store variables from last run for plotting
            SignatureNoise_Last = None
            Distance_Last = None

            for r in range(Nr):
                # Generate new noise
                SignatureNoise = SignatureBlurred + SigmaNoise * np.random.randn(
                    NSymbols, N
                )

                # Calculate Distances
                Dist_r = np.linalg.norm(
                    Signature[:, np.newaxis, :] - SignatureNoise[np.newaxis, :, :],
                    axis=2,
                )

                # Classify
                Predicted_r = np.argmin(Dist_r, axis=0)

                # Accumulate Confusion
                for j in range(NSymbols):  # True class j
                    ConfusionMatrix[j, Predicted_r[j]] += 1

                # Accumulate Errors
                TotalErrors += np.sum(Predicted_r != np.arange(NSymbols))

                # Keep last run for display
                if r == Nr - 1:
                    Distance_Last = Dist_r
                    SignatureNoise_Last = SignatureNoise

            # Average Error Rate
            ErrorRate = 100 * TotalErrors / (Nr * NSymbols)

            # Display (Using last run for signals/dist)
            SymbolList = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

            # Figure 1: Signatures (Last Run)
            plt.figure(figsize=(12, 8))
            for i in range(NSymbols):
                plt.subplot(3, 4, i + 1)
                plt.plot(k, Signature[i], "r", label="Clean")
                plt.plot(k, SignatureBlurred[i], "g", label="Blurred")
                plt.plot(k, SignatureNoise_Last[i], "b", label="Noisy")
                plt.title(SymbolList[i])
                plt.grid(True)
                plt.ylim([np.min(SignatureNoise_Last), np.max(SignatureNoise_Last)])

            plt.tight_layout()
            plt.savefig("Figure1311.png")

            # Figure 2: Distance stems (Last Run)
            plt.figure(figsize=(12, 8))
            for i in range(NSymbols):
                plt.subplot(3, 4, i + 1)
                plt.stem(range(NSymbols), Distance_Last[i])
                plt.title(f"Dist. From {SymbolList[i]}")
                plt.xlabel("to")
                plt.grid(True)

            plt.tight_layout()
            plt.savefig("Figure1311Bis.png")

            # Figure 3: Distance Matrix (Gray) (Last Run)
            plt.figure()
            plt.imshow(Distance_Last, interpolation="nearest", cmap="gray")
            plt.colorbar()
            plt.title("Distance Matrix (Last Run)")
            plt.xlabel("Noisy Sample Index")
            plt.ylabel("Prototype Index")
            plt.savefig("Figure1311Ter.png")

            # Figure 4: Accumulated Confusion Matrix (Gray Tones!)
            plt.figure()
            plt.imshow(ConfusionMatrix, interpolation="nearest", cmap="Greys")
            plt.colorbar(label="Count (out of 100)")
            plt.title(f"Confusion Matrix ({Nr} runs, Mean Error = {ErrorRate:.1f}%)")
            plt.xlabel("Predicted Class")
            plt.ylabel("True Class")

            # Add grid/ticks for clarity
            plt.xticks(np.arange(NSymbols), SymbolList)
            plt.yticks(np.arange(NSymbols), SymbolList)

            plt.savefig("Figure1311Quater.png")

            print(f"Mean Error Rate ({Nr} runs): {ErrorRate:.1f}%")
            print("Confusion Matrix saved to Figure1311Quater.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1313(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1313.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.feature import match_template
            import ia870 as ia
            from helpers.data_path import dip_data

            # Data
            path_f = dip_data("Fig1209(a)(Hurricane Andrew).tif")
            path_h = dip_data("Fig1209(b)(eye template).tif")
            f = imread(path_f)
            h = imread(path_h)

            # Correlation
            Corr = match_template(f, h)

            # Find max
            ij = np.unravel_index(np.argmax(Corr), Corr.shape)
            RowMaxCorr, ColMaxCorr = ij[0], ij[1]  # Top-left coordinates of the match

            # MaxCorrImg
            MaxCorrImg = np.zeros_like(f, dtype=bool)

            # We can mark the center of the template for better visualization
            h_h, h_w = h.shape
            center_r = RowMaxCorr + h_h // 2
            center_c = ColMaxCorr + h_w // 2

            # Ensure bounds
            center_r = min(max(center_r, 0), f.shape[0] - 1)
            center_c = min(max(center_c, 0), f.shape[1] - 1)

            MaxCorrImg[center_r, center_c] = 1

            # Display
            fig = plt.figure(figsize=(10, 8))
            fig.canvas.manager.set_window_title("Correlation")

            # 1. Image f
            plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f")
            plt.axis("off")

            # 2. Template h
            plt.subplot(2, 2, 2)
            plt.imshow(h, cmap="gray")
            plt.title("h = Template")
            plt.axis("off")

            # 3. Correlation surface
            plt.subplot(2, 2, 3)
            plt.imshow(Corr, cmap="gray")
            plt.title("r_{fh}")
            plt.colorbar()
            plt.axis("off")

            # 4. Max dilated
            marker_dilated = ia.iadil(MaxCorrImg, ia.iasecross(5))

            plt.subplot(2, 2, 4)
            plt.imshow(marker_dilated, cmap="gray")
            plt.title("max(r_{fh})")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure1313.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1315surf(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1315SURF.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.feature import ORB, match_descriptors
            import os
            from helpers.data_path import dip_data

            try:
                from skimage.feature import plot_matches as _plot_matches_legacy

                def _plot_matches_compat(ax, image0, image1, keypoints0, keypoints1, matches):
                    return _plot_matches_legacy(
                        ax,
                        image0,
                        image1,
                        keypoints0,
                        keypoints1,
                        matches,
                        only_matches=True,
                    )

            except Exception:
                from skimage.feature import plot_matched_features as _plot_matched_features

                def _plot_matches_compat(ax, image0, image1, keypoints0, keypoints1, matches):
                    return _plot_matched_features(
                        image0,
                        image1,
                        keypoints0=keypoints0,
                        keypoints1=keypoints1,
                        matches=matches,
                        ax=ax,
                        only_matches=True,
                    )

            def Figure1315SURF():
                """Figure1315SURF."""
                # Init

                # Data
                path_f = dip_data("circuitboard.tif")
                path_pattern = dip_data("circuitboard-connector.tif")

                if not os.path.exists(path_f) or not os.path.exists(path_pattern):
                    print("Warning: Images not found at hardcoded path.")

                # Load as grayscale
                f = imread(path_f, as_gray=True)
                pattern = imread(path_pattern, as_gray=True)

                # Feature Detection using ORB (Oriented FAST and Rotated BRIEF)
                # skimage 0.18.3 does not generally contain SIFT/SURF.
                # ORB is a good alternative for rotation invariant matching.

                descriptor_extractor = ORB(n_keypoints=1000)

                # Detect and Extract f
                descriptor_extractor.detect_and_extract(f)
                keypoints_f = descriptor_extractor.keypoints
                descriptors_f = descriptor_extractor.descriptors

                # Detect and Extract Pattern
                descriptor_extractor.detect_and_extract(pattern)
                keypoints_pattern = descriptor_extractor.keypoints
                descriptors_pattern = descriptor_extractor.descriptors

                print("Algorithm: ORB (skimage)")
                print(f"Points in f: {len(keypoints_f)}")
                print(f"Points in Pattern: {len(keypoints_pattern)}")

                # Matching
                # binary descriptors use hamming distance by default? match_descriptors handles this.
                matches12 = match_descriptors(
                    descriptors_f, descriptors_pattern, cross_check=True
                )

                print(f"Matches found: {len(matches12)}")

                # Display

                # 1. Keypoints
                plt.figure(figsize=(12, 10))

                plt.subplot(2, 2, 1)
                plt.imshow(f, cmap="gray")
                plt.title("f")
                plt.axis("off")

                plt.subplot(2, 2, 2)
                plt.imshow(pattern, cmap="gray")
                plt.title("Pattern")
                plt.axis("off")

                plt.subplot(2, 2, 3)
                plt.imshow(f, cmap="gray")
                plt.plot(keypoints_f[:, 1], keypoints_f[:, 0], ".r", markersize=2)
                plt.title(f"f, N = {len(keypoints_f)} (ORB)")
                plt.axis("off")

                plt.subplot(2, 2, 4)
                plt.imshow(pattern, cmap="gray")
                plt.plot(
                    keypoints_pattern[:, 1], keypoints_pattern[:, 0], ".r", markersize=2
                )
                plt.title(f"Pattern, N = {len(keypoints_pattern)} (ORB)")
                plt.axis("off")

                plt.tight_layout()
                plt.savefig("Figure1315SURF.png")

                # 2. Matches
                plt.figure(figsize=(14, 6))
                ax = plt.gca()

                # Show top 50 matches if too many? ORB usually produces many.
                # matches12 indices are sorted by distance? No guaranteed.
                # match_descriptors returns indices.

                _plot_matches_compat(
                    ax,
                    f,
                    pattern,
                    keypoints_f,
                    keypoints_pattern,
                    matches12,
                )
                plt.title(f"Candidate point matches ({len(matches12)} total)")
                plt.axis("off")
                plt.savefig("Figure1315SURFBis.png")

                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1320(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1320.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.20 - Minimum distance / Gaussian classifier in 3D."""

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
                X1 = 0.2 * np.random.randn(3, 10) + np.tile(
                    np.array([[1.0], [1.0], [1.0]]), (1, 10)
                )
                X2 = 0.2 * np.random.randn(3, 10) + np.tile(
                    np.array([[-1.0], [-1.0], [-1.0]]), (1, 10)
                )
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

            print(
                "d1(X) =",
                f"{w1[0, 0]:.3f}*x1 + {w1[1, 0]:.3f}*x2 + {w1[2, 0]:.3f}*x3 + {b1:.3f}",
            )
            print(
                "d2(X) =",
                f"{w2[0, 0]:.3f}*x1 + {w2[1, 0]:.3f}*x2 + {w2[2, 0]:.3f}*x3 + {b2:.3f}",
            )
            print(
                "d12(X)=",
                f"{w[0, 0]:.3f}*x1 + {w[1, 0]:.3f}*x2 + {w[2, 0]:.3f}*x3 + {b:.3f}",
            )

            # Display
            fig = plt.figure(1, figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            # Points and means
            ax.scatter(
                X1[0, :],
                X1[1, :],
                X1[2, :],
                marker="o",
                s=90,
                edgecolors="k",
                facecolors="r",
            )
            ax.scatter(
                m1[0, 0],
                m1[1, 0],
                m1[2, 0],
                marker="s",
                s=90,
                edgecolors="k",
                facecolors="r",
            )

            ax.scatter(
                X2[0, :],
                X2[1, :],
                X2[2, :],
                marker="o",
                s=90,
                edgecolors="k",
                facecolors="g",
            )
            ax.scatter(
                m2[0, 0],
                m2[1, 0],
                m2[2, 0],
                marker="s",
                s=90,
                edgecolors="k",
                facecolors="g",
            )

            # Decision plane: w1*x + w2*y + w3*z + b = 0 -> z = -(w1*x + w2*y + b)/w3
            les_x, les_y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
            if abs(float(w[2, 0])) > 1e-12:
                les_z = -(float(w[0, 0]) * les_x + float(w[1, 0]) * les_y + b) / float(
                    w[2, 0]
                )
                color = 0.5 * np.ones((3, 3, 3), dtype=float)
                ax.plot_surface(
                    les_x, les_y, les_z, facecolors=color, shade=False, alpha=0.8
                )

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")
            ax.set_title(
                "X_1 (or), X_2 (og), μ_1 (sr), μ_2 (sg), X / d(X, X_1) == d(X, X_2) (k)"
            )
            ax.grid(True)

            # MATLAB axis comment is optional; keep auto for choix=2 random case.

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1320.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()

            # -----------------------------------------------------------------------------
            # LaTeX-style outputs (console)
            # -----------------------------------------------------------------------------
            def to_latex_matrix(A: np.ndarray) -> str:
                """to_latex_matrix."""
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
            print(
                "latex(d1):",
                f"{w1[0, 0]:.3f} x_1 + {w1[1, 0]:.3f} x_2 + {w1[2, 0]:.3f} x_3 + {b1:.3f}",
            )
            print(
                "latex(d2):",
                f"{w2[0, 0]:.3f} x_1 + {w2[1, 0]:.3f} x_2 + {w2[2, 0]:.3f} x_3 + {b2:.3f}",
            )
            print(
                "latex(d12):",
                f"{w[0, 0]:.3f} x_1 + {w[1, 0]:.3f} x_2 + {w[2, 0]:.3f} x_3 + {b:.3f}",
            )
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1321(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1321.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image

            from helpers.imstack2vectors import imstack2vectors
            from helpers.covmatrix import covmatrix
            from helpers.bayesgauss import bayesgauss
            from helpers.data_path import dip_data

            print("Running Figure1321 (Bayes classification of remote data)...")

            def _imread_gray(path: str) -> np.ndarray:
                """_imread_gray."""
                arr = np.asarray(Image.open(path))
                if arr.ndim == 3:
                    arr = arr[..., 0]
                return arr

            # Read masks
            B1 = _imread_gray(dip_data("WashingtonDC-mask-water-512.tif"))
            B2 = _imread_gray(dip_data("WashingtonDC-mask-urban-512.tif"))
            B3 = _imread_gray(dip_data("WashingtonDC-mask-vegetation-512.tif"))

            # Read multispectral images
            f1 = _imread_gray(dip_data("WashingtonDC-Band1-Blue-512.tif"))
            f2 = _imread_gray(dip_data("WashingtonDC-Band2-Green-512.tif"))
            f3 = _imread_gray(dip_data("WashingtonDC-Band3-Red-512.tif"))
            f4 = _imread_gray(dip_data("WashingtonDC-Band4-NearInfrared-512.tif"))

            # Stack images
            stack = np.dstack((f1, f2, f3, f4))

            # Extract pattern vectors in mask regions
            X1, R1 = imstack2vectors(stack, B1)
            X2, R2 = imstack2vectors(stack, B2)
            X3, R3 = imstack2vectors(stack, B3)

            # Training patterns (odd rows in MATLAB -> Python step 2 from index 0)
            T1 = X1[0::2, :]
            T2 = X2[0::2, :]
            T3 = X3[0::2, :]

            # Mean vectors and covariance matrices
            C1, m1 = covmatrix(T1)
            C2, m2 = covmatrix(T2)
            C3, m3 = covmatrix(T3)

            # Arrays for bayesgauss
            CA = np.dstack((C1, C2, C3))
            MA = np.hstack((m1, m2, m3)).T

            # Classify training set
            dT1 = bayesgauss(T1, CA, MA)
            dT2 = bayesgauss(T2, CA, MA)
            dT3 = bayesgauss(T3, CA, MA)

            # Training counts
            Class1_to_1_T = int(np.sum(dT1 == 1))
            Class1_to_2_T = int(np.sum(dT1 == 2))
            Class1_to_3_T = int(np.sum(dT1 == 3))
            Class2_to_1_T = int(np.sum(dT2 == 1))
            Class2_to_2_T = int(np.sum(dT2 == 2))
            Class2_to_3_T = int(np.sum(dT2 == 3))
            Class3_to_1_T = int(np.sum(dT3 == 1))
            Class3_to_2_T = int(np.sum(dT3 == 2))
            Class3_to_3_T = int(np.sum(dT3 == 3))

            # Independent data (even rows in MATLAB -> Python step 2 from index 1)
            I1 = X1[1::2, :]
            I2 = X2[1::2, :]
            I3 = X3[1::2, :]

            # Classify testing set
            dI1 = bayesgauss(I1, CA, MA)
            dI2 = bayesgauss(I2, CA, MA)
            dI3 = bayesgauss(I3, CA, MA)

            # Testing counts
            Class1_to_1 = int(np.sum(dI1 == 1))
            Class1_to_2 = int(np.sum(dI1 == 2))
            Class1_to_3 = int(np.sum(dI1 == 3))
            Class2_to_1 = int(np.sum(dI2 == 1))
            Class2_to_2 = int(np.sum(dI2 == 2))
            Class2_to_3 = int(np.sum(dI2 == 3))
            Class3_to_1 = int(np.sum(dI3 == 1))
            Class3_to_2 = int(np.sum(dI3 == 2))
            Class3_to_3 = int(np.sum(dI3 == 3))

            print("Training confusion-like counts:")
            print(
                f"Class1 -> [1,2,3] = [{Class1_to_1_T}, {Class1_to_2_T}, {Class1_to_3_T}]"
            )
            print(
                f"Class2 -> [1,2,3] = [{Class2_to_1_T}, {Class2_to_2_T}, {Class2_to_3_T}]"
            )
            print(
                f"Class3 -> [1,2,3] = [{Class3_to_1_T}, {Class3_to_2_T}, {Class3_to_3_T}]"
            )
            print("Testing confusion-like counts:")
            print(f"Class1 -> [1,2,3] = [{Class1_to_1}, {Class1_to_2}, {Class1_to_3}]")
            print(f"Class2 -> [1,2,3] = [{Class2_to_1}, {Class2_to_2}, {Class2_to_3}]")
            print(f"Class3 -> [1,2,3] = [{Class3_to_1}, {Class3_to_2}, {Class3_to_3}]")

            # Classify all pixels in whole image region
            B = np.ones_like(f1)
            X, R = imstack2vectors(stack, B)
            dAll = bayesgauss(X, CA, MA)

            # Rebuild class images using MATLAB/Fortran linear indexing order.
            M, N = f1.shape
            class1_vec = np.zeros(M * N, dtype=np.uint8)
            class2_vec = np.zeros(M * N, dtype=np.uint8)
            class3_vec = np.zeros(M * N, dtype=np.uint8)

            class1_vec[np.where(dAll == 1)[0]] = 1
            class2_vec[np.where(dAll == 2)[0]] = 1
            class3_vec[np.where(dAll == 3)[0]] = 1

            class1 = np.reshape(class1_vec, (M, N), order="F")
            class2 = np.reshape(class2_vec, (M, N), order="F")
            class3 = np.reshape(class3_vec, (M, N), order="F")

            # Display
            fig = plt.figure(1)

            plt.subplot(3, 3, 1)
            plt.imshow(f1, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 2)
            plt.imshow(f2, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 3)
            plt.imshow(f3, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 4)
            plt.imshow(f4, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 5)
            plt.imshow((B1 != 0) | (B2 != 0) | (B3 != 0), cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 6)
            plt.axis("off")

            plt.subplot(3, 3, 7)
            plt.imshow(class1, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 8)
            plt.imshow(class2, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 9)
            plt.imshow(class3, cmap="gray")
            plt.axis("off")

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1321.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1324(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1324.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.24 - Perceptron example for patterns in Fig. 13.22."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from DIP4eFigures.perceptronTraining4e import perceptronTraining4e

            print("Running Figure1324 (perceptron example)...")

            # Data: augmented patterns (columns)
            X = np.array([[1, 3], [1, 3], [1, 1]], dtype=float)
            r = np.array([-1, 1], dtype=float)
            w0 = np.array([0, 0, 0], dtype=float)

            # Learning
            w, epochs = perceptronTraining4e(X, r, alpha=1.0, nepochs=100, w0=w0)
            print(f"Converged in {epochs} epochs to w = {w}")

            # Decision surface shown intersecting the xy-plane.
            # MATLAB: [x2, x1] = meshgrid(0:.1:4, 0:.1:4)
            x2, x1 = np.meshgrid(
                np.arange(0, 4.0 + 0.1, 0.1), np.arange(0, 4.0 + 0.1, 0.1)
            )

            # Decision function d(x,y) = w1*x + w2*y + w3
            d = w[0] * x1 + w[1] * x2 + w[2]
            print(f"Decision boundary: {w[0]:.3f}*x + {w[1]:.3f}*y + {w[2]:.3f} = 0")

            # Display
            fig = plt.figure(1, figsize=(11, 5))

            # Subplot 1: points + implicit decision boundary
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(X[0, 0], X[1, 0], "or")
            ax1.plot(X[0, 1], X[1, 1], "og")

            xx = np.linspace(0, 4, 400)
            if abs(w[1]) > 1e-12:
                yy = -(w[0] * xx + w[2]) / w[1]
                ax1.plot(xx, yy, "k-")
            elif abs(w[0]) > 1e-12:
                x_const = -w[2] / w[0]
                ax1.plot([x_const, x_const], [0, 4], "k-")

            ax1.set_xlim(0, 4)
            ax1.set_ylim(0, 4)
            ax1.set_aspect("equal", adjustable="box")

            # Subplot 2: decision surface + constant plane
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            # Use the same physical coordinate domain as subplot 1 for a clearer view.
            surf_color = np.ones((41, 41, 3), dtype=float) * 0.75
            ax2.plot_surface(
                x1,
                x2,
                d,
                facecolors=surf_color,
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=0.7,
            )

            C = 0.205 * float(np.max(d))
            plane = np.zeros_like(x1, dtype=float) + C
            ax2.plot_surface(
                x1,
                x2,
                plane,
                color="#9ecae1",
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=0.95,
            )

            ax2.view_init(elev=28, azim=-55)
            ax2.grid(False)
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_box_aspect((1, 1, 0.65))

            # Save
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1324.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1324lmse(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1324LMSE.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            def Figure1324LMSE():
                """Figure1324LMSE."""
                # Parameters
                MaxIter = 250
                Alpha = 0.1

                # Data from C1
                x1 = np.array([3, 3])
                y1 = np.array([3, 3, 1])
                r1 = 1

                # Data from C2
                x2 = np.array([1, 1])
                y2 = np.array([1, 1, 1])
                r2 = -1

                # Weight update
                # MATLAB: LesW = zeros(3, MaxIter)
                LesW = np.zeros((3, MaxIter))

                # MATLAB: repmat([y1, y2], 1, MaxIter) -> Alternating y1, y2
                # In Python, we can just alternate in the loop or construct the array
                # Since MaxIter is 250, we have 250 training steps?
                # MATLAB loop: for iter = 1 : MaxIter-1
                # Accesses LesY(:, iter)
                # LesY has width 2*MaxIter technically if we simply repmat?
                # MATLAB: repmat([y1, y2], 1, MaxIter).
                # [y1, y2] is 3x2.
                # Repmat 1xMaxIter -> Result is 3 x (2*MaxIter).
                # The loop goes up to MaxIter-1. So it uses the first MaxIter columns.
                # Column 1: y1. Column 2: y2. Column 3: y1. ...

                LesY_cols = []
                Lesr_vals = []
                for _ in range(MaxIter):  # Enough to cover MaxIter
                    LesY_cols.append(y1)
                    Lesr_vals.append(r1)
                    LesY_cols.append(y2)
                    Lesr_vals.append(r2)

                LesY = np.array(LesY_cols).T  # (3, 2*MaxIter)
                Lesr = np.array(Lesr_vals)  # (2*MaxIter,)

                # Learning Loop
                for iter_idx in range(MaxIter - 1):  # 0 to MaxIter-2
                    # Current weights: LesW[:, iter_idx]
                    w_curr = LesW[:, iter_idx]
                    y_curr = LesY[:, iter_idx]
                    r_curr = Lesr[iter_idx]

                    # Output = dot (LesW(:, iter), LesY(:, iter));
                    output = np.dot(w_curr, y_curr)

                    # Error = Lesr(iter) - Output;
                    error = r_curr - output

                    # LesW (:, iter+1) = LesW (:, iter) + Alpha * Error * LesY(:, iter);
                    w_next = w_curr + Alpha * error * y_curr
                    LesW[:, iter_idx + 1] = w_next

                # Decision boundary (Final weights)
                w_final = LesW[:, -1]
                print(f"Final Weights: {w_final}")

                # Display
                fig = plt.figure(figsize=(12, 10))

                # Plot weights evolution
                titles = [
                    f"w_1, w_1^* = {w_final[0]:.3f}",
                    f"w_2, w_2^* = {w_final[1]:.3f}",
                    f"w_3, w_3^* = {w_final[2]:.3f}",
                ]

                for i in range(3):
                    plt.subplot(2, 2, i + 1)
                    plt.plot(LesW[i, :])
                    plt.xlabel("Iter")
                    plt.title(titles[i])
                    plt.grid(True)
                    # axis tight equivalent?
                    plt.autoscale(enable=True, axis="x", tight=True)

                # Plot Decision Boundary
                plt.subplot(2, 2, 4)
                plt.plot(x1[0], x1[1], "ok", markersize=8, label="C1 (3,3)")
                plt.plot(x2[0], x2[1], "ok", markersize=8, label="C2 (1,1)")

                # Decision line: w1*x + w2*y + w3 = 0
                # y = (-w1*x - w3) / w2
                # Define range for x
                x_range = np.linspace(0, 4, 100)

                if abs(w_final[1]) > 1e-6:
                    y_range = (-w_final[0] * x_range - w_final[2]) / w_final[1]
                    plt.plot(x_range, y_range, "-b", label="Decision Boundary")
                else:
                    # Vertical line x = -w3 / w1
                    if abs(w_final[0]) > 1e-6:
                        x_line = -w_final[2] / w_final[0]
                        plt.axvline(x=x_line, color="b", label="Decision Boundary")

                plt.title("Decision boundary")
                plt.axis([0, 4, 0, 4])
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.savefig("Figure1324LMSE.png")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1324perceptron(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1324Perceptron.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            def Figure1324Perceptron():
                """Figure1324Perceptron."""
                # Parameters
                NbRep = 12
                Alpha = 1

                # Data from C1
                x1 = np.array([3, 3])
                y1 = np.array([3, 3, 1])

                # Data from C2
                x2 = np.array([1, 1])
                y2 = np.array([1, 1, 1])

                # Weight update
                MaxIter = NbRep * 2
                LesWeights = np.zeros(
                    (3, MaxIter + 1)
                )  # +1 to match MATLAB indexing/storage (iter+1)

                # LesY: Alternating [y1, y2, y1, y2...]
                # MATLAB: repmat([y1, y2], 1, NbRep)
                # y1 is col vector in MATLAB.
                # In Python, we construct the sequence.
                LesY_list = []
                for _ in range(NbRep):
                    LesY_list.append(y1)
                    LesY_list.append(y2)
                LesY = np.array(LesY_list).T  # (3, 2*NbRep)

                # Check shape
                # Expected: (3, 2*12) = (3, 24)

                LesOutput = np.zeros((2, MaxIter + 1))
                # Output = np.zeros(MaxIter + 1)

                for iter_idx in range(MaxIter):  # 0 to 23
                    w_curr = LesWeights[:, iter_idx]
                    y_curr = LesY[:, iter_idx]

                    # Check misclassification
                    dot_prod = np.dot(w_curr, y_curr)

                    # Identify if y_curr is y1 or y2
                    is_y1 = np.array_equal(y_curr, y1)
                    is_y2 = np.array_equal(y_curr, y2)

                    w_next = w_curr.copy()

                    if is_y1 and dot_prod <= 0:
                        # Class 1 misclassified (should be > 0)
                        w_next = w_curr + Alpha * y_curr
                    elif is_y2 and dot_prod >= 0:
                        # Class 2 misclassified (should be < 0)
                        w_next = w_curr - Alpha * y_curr
                    else:
                        # Correctly classified
                        w_next = w_curr

                    LesWeights[:, iter_idx + 1] = w_next

                    # Log outputs for y1 and y2 with NEW weights (as per MATLAB code lines 44-45)
                    LesOutput[0, iter_idx + 1] = np.dot(w_next, y1)
                    LesOutput[1, iter_idx + 1] = np.dot(w_next, y2)

                # Decision boundary
                w_final = LesWeights[:, -1]  # Last column (index MaxIter)
                print(f"Final Weights: {w_final}")

                # Display
                fig = plt.figure(figsize=(12, 10))

                # Plot weights evolution
                titles = ["w_1", "w_2", "w_3"]
                for i in range(3):
                    plt.subplot(2, 2, i + 1)
                    # MATLAB: stem(LesWeights(i, :))
                    # Remove last column? MATLAB loop went to MaxIter, stored in Iter+1.
                    # So LesWeights has MaxIter+1 columns.
                    plt.stem(LesWeights[i, :], use_line_collection=True)
                    plt.xlabel("Iter")
                    plt.title(titles[i])
                    plt.autoscale(enable=True, axis="x", tight=True)
                    plt.grid(True)

                # Plot Decision Boundary
                plt.subplot(2, 2, 4)
                plt.plot(x1[0], x1[1], "ok", markersize=8, label="C1 (3,3)")
                plt.plot(x2[0], x2[1], "ok", markersize=8, label="C2 (1,1)")

                # Decision line: w1*x + w2*y + w3 = 0
                x_range = np.linspace(0, 4, 100)

                if abs(w_final[1]) > 1e-6:
                    y_range = (-w_final[0] * x_range - w_final[2]) / w_final[1]
                    plt.plot(x_range, y_range, "-b", label="Decision Boundary")
                else:
                    if abs(w_final[0]) > 1e-6:
                        x_line = -w_final[2] / w_final[0]
                        plt.axvline(x=x_line, color="b", label="Decision Boundary")

                plt.title("Decision boundary")
                plt.axis([0, 3, 0, 3])
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.savefig("Figure1324Perceptron.png")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1325(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1325.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.25 - Quadratic functions for 1 and 2 variables."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt

            print("Running Figure1325 (quadratic functions)...")

            # Plot of 1D quadratic function.
            x = np.arange(0.0, 2.0 + 0.01, 0.01)
            r = 1.0
            E = 0.5 * ((r - x) ** 2)

            # Plot of 2D quadratic function.
            w2, w1 = np.meshgrid(
                np.arange(0.0, 2.0 + 0.005, 0.005), np.arange(0.0, 2.0 + 0.01, 0.01)
            )
            f = 0.5 * ((w1 - 1.0) ** 2 + (w2 - 1.0) ** 2)

            # Display
            fig = plt.figure(1, figsize=(10, 4.5))

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(E, "k-")
            ax1.set_box_aspect(1)
            ax1.autoscale(enable=True, axis="both", tight=True)

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            row_idx = slice(0, f.shape[0], 8)
            col_idx = slice(0, f.shape[1], 8)
            X = np.arange(f[row_idx, col_idx].shape[1])
            Y = np.arange(f[row_idx, col_idx].shape[0])
            Xg, Yg = np.meshgrid(X, Y)
            ax2.plot_wireframe(Xg, Yg, f[row_idx, col_idx], color="k", linewidth=0.7)
            ax2.set_box_aspect((1, 1, 0.6))

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1325.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1326(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1326.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.26 - LMSE perceptron error plots for separable and inseparable Iris data."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.io import loadmat

            from DIP4eFigures.lmsePerceptronTraining4e import lmsePerceptronTraining4e
            from DIP4eFigures.perceptronClassifier4e import perceptronClassifier4e
            from helpers.data_path import dip_data

            print("Running Figure1326 (LMSE perceptron errors on iris data)...")

            MAT_PATH = dip_data("fisheriris.mat")

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
            X_nonsep_aug = np.vstack(
                (X_nonsep, np.ones((1, X_nonsep.shape[1]), dtype=float))
            )
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
                np.asarray(inputNonSep["X"]),
                outputNonSep["W"],
                np.asarray(inputNonSep["R"]),
            )

            print(
                f"Separable set (setosa vs versicolor): errors = {int(numErrorsSep)}, recognition = {recogRateSep:.2f}%"
            )
            print(
                f"Nonseparable set (versicolor vs virginica): errors = {int(numErrorsNonSep)}, recognition = {recogRateNonSep:.2f}%"
            )

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

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1326.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1329(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1329.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.29 - Three activation functions."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt

            print("Running Figure1329 (activation functions)...")

            # (a) Sigmoid
            z = np.arange(-6.0, 6.0 + 0.01, 0.01)
            hs = 1.0 / (1.0 + np.exp(-z))

            # (b) tanh
            ht = np.tanh(z)

            # (c) ReLU
            hReLU = np.maximum(0.0, z)

            # Display
            fig = plt.figure(1, figsize=(12, 4))

            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(hs, "k-")
            ax1.set_xlim([0, 1200])
            ax1.set_ylim([0, 1])
            ax1.set_box_aspect(1)

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.plot(ht, "k-")
            ax2.set_xlim([0, 1200])
            ax2.set_ylim([-1, 1])
            ax2.set_box_aspect(1)

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(hReLU, "k-")
            ax3.set_xlim([0, 1200])
            ax3.set_ylim([0, 6])
            ax3.set_box_aspect(1)

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1329.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1334(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1334.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.34 - XOR gate using neuralNet4e and DIP4e helpers."""

            import numpy as np
            import matplotlib.pyplot as plt

            from DIP4eFigures.neuralNet4e import neuralNet4e
            from DIP4eFigures.moreTrainingPatterns4e import moreTrainingPatterns4e
            from DIP4eFigures.patternShuffle4e import patternShuffle4e

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
                "W": [
                    None,
                    None,
                    np.random.rand(HiddenSizes, 2),
                    np.random.rand(2, HiddenSizes),
                ],
                "b": [None, None, np.random.rand(HiddenSizes, 1), np.random.rand(2, 1)],
                "Correction": Correction,
            }

            Inputt = {"X": Input, "R": Target}
            Inputt["X"], Inputt["R"] = moreTrainingPatterns4e(
                Inputt["X"], Inputt["R"], NRep
            )
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
                "X": np.vstack(
                    (X.reshape(1, -1, order="F"), Y.reshape(1, -1, order="F"))
                )
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

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1334.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1335(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1335.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt

            def sigmoid(x: Any):
                """sigmoid."""
                return 1 / (1 + np.exp(-x))

            class SimpleMLP:
                def __init__(
                    self,
                    input_size: Any,
                    hidden_size: Any,
                    output_size: Any,
                    learning_rate: Any = 0.02,
                    seed: Any = None,
                ):
                    """__init__."""
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.output_size = output_size
                    self.learning_rate = learning_rate

                    if seed is not None:
                        np.random.seed(seed)

                    self.W1 = np.random.randn(hidden_size, input_size)
                    self.b1 = np.random.randn(hidden_size, 1)
                    self.W2 = np.random.randn(output_size, hidden_size)
                    self.b2 = np.random.randn(output_size, 1)

                def forward(self, X: Any):
                    """forward."""
                    self.z1 = np.dot(self.W1, X) + self.b1
                    self.a1 = sigmoid(self.z1)
                    self.z2 = np.dot(self.W2, self.a1) + self.b2
                    self.a2 = sigmoid(self.z2)
                    return self.a2

                def train(self, X: Any, Target: Any, epochs: Any = 1000):
                    """train."""
                    costs = []
                    N_samples = X.shape[1]

                    for epoch in range(epochs):
                        perm = np.random.permutation(N_samples)
                        X_shuffled = X[:, perm]
                        T_shuffled = Target[:, perm]

                        epoch_cost = 0

                        for i in range(N_samples):
                            x = X_shuffled[:, i : i + 1]
                            t = T_shuffled[:, i : i + 1]

                            output = self.forward(x)
                            error = t - output
                            epoch_cost += np.sum(error**2)

                            delta2 = error * (output * (1 - output))
                            delta1 = np.dot(self.W2.T, delta2) * (
                                self.a1 * (1 - self.a1)
                            )

                            self.W2 += self.learning_rate * np.dot(delta2, self.a1.T)
                            self.b2 += self.learning_rate * delta2
                            self.W1 += self.learning_rate * np.dot(delta1, x.T)
                            self.b1 += self.learning_rate * delta1

                        costs.append(epoch_cost / N_samples)
                    return costs

            def Figure1335():
                """Figure1335."""
                # Data columns: [1,1], [-1,-1], [-1,1], [1,-1]
                Input = np.array([[1, -1, -1, 1], [1, -1, 1, -1]])

                # Class 1: [1,1], [-1,-1] -> Target [1, 0] ? MATLAB: [1, 1, 0, 0] (Rows 1 and 2?)
                # MATLAB Target: [1, 1, 0, 0; 0, 0, 1, 1]
                # Sample 1 (1,1): [1;0] -> Class 1
                # Sample 2 (-1,-1): [1;0] -> Class 1
                # Sample 3 (-1,1): [0;1] -> Class 2
                # Sample 4 (1,-1): [0;1] -> Class 2
                Target = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])

                # Augment Data
                NRep = 100
                Input_Aug = np.tile(Input, (1, NRep))
                Target_Aug = np.tile(Target, (1, NRep))

                best_mlp = None
                best_costs = []
                min_mse = float("inf")

                # Retry Loop
                for attempt in range(10):
                    print(f"Attempt {attempt + 1}...")
                    mlp = SimpleMLP(2, 2, 2, learning_rate=0.05, seed=attempt)
                    costs = mlp.train(
                        Input_Aug, Target_Aug, epochs=500
                    )  # 500 epochs sufficient?
                    final_mse = costs[-1]
                    print(f"  MSE: {final_mse:.4f}")

                    if final_mse < min_mse:
                        min_mse = final_mse
                        best_mlp = mlp
                        best_costs = costs

                    if final_mse < 0.05:
                        print("  Converged!")
                        break

                if best_mlp is None:
                    print("Warning: Did not converge. Using best attempt.")
                    best_mlp = mlp
                    best_costs = costs

                # Testing
                OutputTest = best_mlp.forward(Input)

                # Visualization
                fig = plt.figure(figsize=(10, 12))

                # 1. Target Visualization
                plt.subplot(2, 2, 1)
                for i in range(4):
                    # Class 1 (Target[0] > 0.5) -> Green? MATLAB comments say:
                    # XOR=1 (g), XOR=0 (r).
                    # In MATLAB Target:
                    # Col 1 (1,1) -> [1;0]. XOR(1,1)=0. Wait, 1 XOR 1 is 0.
                    # Col 2 (-1,-1) -> [1;0]. (-1) XOR (-1) is 0.
                    # Col 3 (-1,1) -> [0;1]. (-1) XOR 1 is 1.
                    # Col 4 (1,-1) -> [0;1]. 1 XOR (-1) is 1.
                    # So [1;0] is XOR=0 (Red). [0;1] is XOR=1 (Green).
                    # Check MATLAB logic:
                    # if Target(1, iter) > 0.5 (i.e. Class 1 / XOR=0) -> plot 'or' (Red)
                    # else (Class 2 / XOR=1) -> plot 'og' (Green)

                    if Target[0, i] > 0.5:
                        plt.plot(
                            Input[0, i],
                            Input[1, i],
                            "or",
                            markersize=10,
                            markerfacecolor="none",
                            markeredgewidth=2,
                        )
                    else:
                        plt.plot(
                            Input[0, i],
                            Input[1, i],
                            "og",
                            markersize=10,
                            markerfacecolor="none",
                            markeredgewidth=2,
                        )

                plt.xlabel("x_1")
                plt.ylabel("x_2")
                plt.title("Target: XOR=1 (g), XOR=0 (r)")
                plt.grid(True)
                plt.xlim(-1.5, 1.5)
                plt.ylim(-1.5, 1.5)

                # 2. ANN Output Visualization
                plt.subplot(2, 2, 2)
                for i in range(4):
                    # if OutputTest(1, i) > 0.5 -> Red
                    if OutputTest[0, i] > 0.5:
                        plt.plot(
                            Input[0, i],
                            Input[1, i],
                            "or",
                            markersize=10,
                            markerfacecolor="none",
                            markeredgewidth=2,
                        )
                    else:
                        plt.plot(
                            Input[0, i],
                            Input[1, i],
                            "og",
                            markersize=10,
                            markerfacecolor="none",
                            markeredgewidth=2,
                        )

                plt.xlabel("x_1")
                plt.ylabel("x_2")
                plt.title("ANN: XOR=1 (g), XOR=0 (r)")
                plt.grid(True)
                plt.xlim(-1.5, 1.5)
                plt.ylim(-1.5, 1.5)

                # 3. MSE
                plt.subplot(2, 1, 2)
                plt.plot(best_costs)
                plt.xlabel("Epochs")
                plt.ylabel("MSE")

                # Calculate Recog Rate
                # Class 1: Node 0 > Node 1. Class 2: Node 1 > Node 0.
                Predictions = np.argmax(OutputTest, axis=0)  # 0 or 1
                Targets = np.argmax(Target, axis=0)  # 0 or 1
                RecogRate = np.mean(Predictions == Targets) * 100

                plt.title(f"MSE Evolution. Recog Rate = {RecogRate:.1f}%")
                plt.grid(True)

                plt.tight_layout()
                plt.savefig("Figure1335.png")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1335usingnntoolbox(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1335UsingNNToolbox.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt

            def sigmoid(x: Any):
                """sigmoid."""
                return 1 / (1 + np.exp(-x))

            class SimpleMLP:
                def __init__(
                    self,
                    input_size: Any,
                    hidden_size: Any,
                    output_size: Any,
                    learning_rate: Any = 0.02,
                    seed: Any = None,
                ):
                    """__init__."""
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.output_size = output_size
                    self.learning_rate = learning_rate

                    if seed is not None:
                        np.random.seed(seed)

                    self.W1 = np.random.randn(hidden_size, input_size)
                    self.b1 = np.random.randn(hidden_size, 1)
                    self.W2 = np.random.randn(output_size, hidden_size)
                    self.b2 = np.random.randn(output_size, 1)

                def forward(self, X: Any):
                    """forward."""
                    self.z1 = np.dot(self.W1, X) + self.b1
                    self.a1 = sigmoid(self.z1)
                    self.z2 = np.dot(self.W2, self.a1) + self.b2
                    self.a2 = sigmoid(self.z2)
                    return self.a2

                def train(self, X: Any, Target: Any, epochs: Any = 1000):
                    """train."""
                    costs = []
                    N_samples = X.shape[1]

                    for epoch in range(epochs):
                        perm = np.random.permutation(N_samples)
                        X_shuffled = X[:, perm]
                        T_shuffled = Target[:, perm]

                        epoch_cost = 0

                        for i in range(N_samples):
                            x = X_shuffled[:, i : i + 1]
                            t = T_shuffled[:, i : i + 1]

                            output = self.forward(x)
                            error = t - output
                            epoch_cost += np.sum(error**2)

                            delta2 = error * (output * (1 - output))
                            delta1 = np.dot(self.W2.T, delta2) * (
                                self.a1 * (1 - self.a1)
                            )

                            self.W2 += self.learning_rate * np.dot(delta2, self.a1.T)
                            self.b2 += self.learning_rate * delta2
                            self.W1 += self.learning_rate * np.dot(delta1, x.T)
                            self.b1 += self.learning_rate * delta1

                        costs.append(epoch_cost / N_samples)
                    return costs

            def Figure1335UsingNNToolbox():
                """Figure1335UsingNNToolbox."""
                # Parameters
                HiddenSizes = 2
                NRep = 30

                # Data
                Input = np.array([[1, -1, -1, 1], [1, -1, 1, -1]])
                Target = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])

                # Data Augmentation (Repmat)
                Inputs = np.tile(Input, (1, NRep))
                Targets = np.tile(Target, (1, NRep))

                # Train
                best_mlp = None
                best_costs = []
                min_mse = float("inf")

                for attempt in range(10):
                    print(f"Attempt {attempt + 1}...")
                    mlp = SimpleMLP(2, 2, 2, learning_rate=0.05, seed=attempt)
                    costs = mlp.train(Inputs, Targets, epochs=500)
                    final_mse = costs[-1]
                    print(f"  MSE: {final_mse:.4f}")

                    if final_mse < min_mse:
                        min_mse = final_mse
                        best_mlp = mlp
                        best_costs = costs

                    if final_mse < 0.05:
                        print("  Converged!")
                        break

                if best_mlp is None:
                    best_mlp = mlp
                    best_costs = costs

                # Output
                OutputRaw = best_mlp.forward(Input)

                # Display 1: Performance (MSE)
                plt.figure(figsize=(8, 6))
                plt.semilogy(best_costs, "b-", linewidth=2, label="Train")
                plt.xlabel("Epochs", fontsize=12, fontweight="bold")
                plt.ylabel("Mean Squared Error (mse)", fontsize=12, fontweight="bold")
                plt.title("Performance (plotperform)", fontsize=14, fontweight="bold")
                plt.grid(True, which="both", ls="-", alpha=0.5)

                # Mark best
                best_epoch = len(best_costs) - 1
                best_val = best_costs[-1]
                plt.plot(
                    best_epoch,
                    best_val,
                    "go",
                    markersize=10,
                    fillstyle="none",
                    markeredgewidth=2,
                    label="Best",
                )
                plt.axvline(x=best_epoch, color="g", linestyle=":", label="_nolegend_")

                plt.legend()
                plt.savefig("Figure1335UsingNNToolbox.png")

                # Display 2: Stems
                fig = plt.figure(figsize=(10, 8))

                plt.subplot(2, 2, 1)
                plt.stem(Input[0, :], use_line_collection=True)
                plt.title("first input")

                plt.subplot(2, 2, 2)
                plt.stem(Input[1, :], use_line_collection=True)
                plt.title("second input")

                # Target in MATLAB plot is flattening 2 outputs into 1D sequence or plotting 2 channels?
                # subplot(2,2,3); stem(Target) implies checking dims.
                # Target is 2x4. stem(Target) usually plots columns as series.
                # To match typical MATLAB stem behavior on matrix: plots each column? Or plots multiple series?
                # Let's plot both Output Nodes.

                plt.subplot(2, 2, 3)
                # Plotting both series
                plt.stem(
                    Target[0, :],
                    linefmt="b-",
                    markerfmt="bo",
                    label="Out1",
                    use_line_collection=True,
                )
                # Offset slightly to see? Or just plot on top.
                # MATLAB's `stem` on a matrix plots columns as separate lines if X is not given?
                # Actually, Target is (2, 4). stem(Target) plots 4 stems, each with 2 values? Or 2 stems of 4?
                # MATLAB treats Matrix as columns. So 4 columns.
                # But usually for time series, we want the sequence.
                # Let's assume we plot the 4 samples.
                # Since we have 2 output nodes, let's plot Node 1 and Node 2.

                # A cleaner Python way:
                x = np.arange(4)
                plt.stem(
                    x - 0.1,
                    Target[0, :],
                    linefmt="b-",
                    markerfmt="bo",
                    label="Class 1",
                    use_line_collection=True,
                )
                plt.stem(
                    x + 0.1,
                    Target[1, :],
                    linefmt="r-",
                    markerfmt="rs",
                    label="Class 2",
                    use_line_collection=True,
                )
                plt.legend()
                plt.title("target")

                plt.subplot(2, 2, 4)
                plt.stem(
                    x - 0.1,
                    OutputRaw[0, :],
                    linefmt="b-",
                    markerfmt="bo",
                    label="Class 1",
                    use_line_collection=True,
                )
                plt.stem(
                    x + 0.1,
                    OutputRaw[1, :],
                    linefmt="r-",
                    markerfmt="rs",
                    label="Class 2",
                    use_line_collection=True,
                )
                plt.title("output")

                plt.tight_layout()
                plt.savefig("Figure1335UsingNNToolboxBis.png")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1336(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1336.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 13.36 - Error plot for XOR problem."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt

            from DIP4eFigures.neuralNet4e import neuralNet4e

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
            output_dir = _os.environ.get(
                "DIP4E_OUTPUT_DIR",
                str(_Path(__file__).resolve().parents[2] / "output"),
            )
            out_path = _os.path.join(output_dir, "Figure1336.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1337_39(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter13 script `Figure1337-39.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            # Figure13.37 and 13.39

            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            import os
            from skimage.io import imread
            from helpers.data_path import dip_data

            def sigmoid(x: Any):
                """sigmoid."""
                # Clip to avoid overflow
                x = np.clip(x, -500, 500)
                return 1 / (1 + np.exp(-x))

            def sigmoid_derivative(x: Any):
                """sigmoid_derivative."""
                s = sigmoid(x)
                return s * (1 - s)

            class SimpleMLP:
                def __init__(
                    self,
                    input_size: Any,
                    hidden_size: Any,
                    output_size: Any,
                    learning_rate: Any = 0.001,
                    seed: Any = None,
                ):
                    """__init__."""
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.output_size = output_size
                    self.learning_rate = learning_rate

                    if seed is not None:
                        np.random.seed(seed)
                    else:
                        np.random.seed(0)

                    # Weights
                    self.W1 = np.random.randn(hidden_size, input_size) * 0.1
                    self.b1 = np.random.randn(hidden_size, 1) * 0.1
                    self.W2 = np.random.randn(output_size, hidden_size) * 0.1
                    self.b2 = np.random.randn(output_size, 1) * 0.1

                def forward(self, X: Any):
                    """forward."""
                    self.z1 = np.dot(self.W1, X) + self.b1
                    self.a1 = sigmoid(self.z1)
                    self.z2 = np.dot(self.W2, self.a1) + self.b2
                    self.a2 = sigmoid(self.z2)
                    return self.a2

                def train(self, X: Any, Target: Any, epochs: Any = 3000):
                    """train."""
                    costs = []
                    N_samples = X.shape[1]

                    for epoch in range(epochs):
                        perm = np.random.permutation(N_samples)
                        X_shuffled = X[:, perm]
                        T_shuffled = Target[:, perm]

                        # Batch or Stochastic? MATLAB's neuralNet4e usually does online/stochastic
                        # But plain Python loops for 4000 samples * 3000 epochs is SLOW.
                        # We will use Mini-Batch (e.g., 32) or Full Batch for speed in Python.
                        # Given "neuralNet4e" iterates patterns, it's stochastic.
                        # To be fast in Python, we'll vectorise.

                        # Forward Full Batch
                        A2 = self.forward(X_shuffled)

                        # Cost
                        error = T_shuffled - A2
                        MSE = np.mean(np.sum(error**2, axis=0))
                        costs.append(MSE)

                        # Backward (Batch)
                        # Delta2
                        delta2 = error * (A2 * (1 - A2))  # (3, N)

                        # Delta1
                        delta1 = np.dot(self.W2.T, delta2) * (
                            self.a1 * (1 - self.a1)
                        )  # (3, N)

                        # Updates
                        # Mean gradient over batch
                        self.W2 += self.learning_rate * np.dot(delta2, self.a1.T)
                        self.b2 += self.learning_rate * np.sum(
                            delta2, axis=1, keepdims=True
                        )
                        self.W1 += self.learning_rate * np.dot(delta1, X_shuffled.T)
                        self.b1 += self.learning_rate * np.sum(
                            delta1, axis=1, keepdims=True
                        )

                        if epoch % 100 == 0:
                            print(f"Epoch {epoch}/{epochs}: MSE={MSE:.4f}")

                    return costs

            # 1. Load Data (fixed absolute paths)
            paths = {
                "VisibleBlue": dip_data("WashingtonDC-Band1-Blue-512.tif"),
                "VisibleGreen": dip_data("WashingtonDC-Band2-Green-512.tif"),
                "VisibleRed": dip_data("WashingtonDC-Band3-Red-512.tif"),
                "NearInfraRed": dip_data("WashingtonDC-Band4-NearInfrared-512.tif"),
                "WaterMask": dip_data("WashingtonDC-mask-water-512.tif"),
                "UrbanMask": dip_data("WashingtonDC-mask-urban-512.tif"),
                "VegetationMask": dip_data("WashingtonDC-mask-vegetation-512.tif"),
            }

            imgs = {}
            for k, p in paths.items():
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing required image: {p}")
                imgs[k] = imread(p)

            # Normalize images
            VB = imgs["VisibleBlue"].astype(float) / 255.0
            VG = imgs["VisibleGreen"].astype(float) / 255.0
            VR = imgs["VisibleRed"].astype(float) / 255.0
            NIR = imgs["NearInfraRed"].astype(float) / 255.0

            # Extract Pixels
            IxWater = np.where(imgs["WaterMask"].flatten())[0]
            IxUrban = np.where(imgs["UrbanMask"].flatten())[0]
            IxVegetation = np.where(imgs["VegetationMask"].flatten())[0]

            def get_features(indices: Any):
                """get_features."""
                return np.vstack(
                    [
                        VB.flat[indices],
                        VG.flat[indices],
                        VR.flat[indices],
                        NIR.flat[indices],
                    ]
                )

            Water = get_features(IxWater)  # (4, N)
            Urban = get_features(IxUrban)
            Vegetation = get_features(IxVegetation)

            # Split Train/Test (50/50)
            def split(arr: Any):
                """split."""
                n = arr.shape[1] // 2
                return arr[:, :n], arr[:, n:]

            Ref_Water, Test_Water = split(Water)
            Ref_Urban, Test_Urban = split(Urban)
            Ref_Vegetation, Test_Vegetation = split(Vegetation)

            # Prepare Training Data
            # X: (4, N_total)
            Train_X = np.hstack([Ref_Water, Ref_Urban, Ref_Vegetation])

            # R: One-hot (3, N_total)
            # Class 1: Water, 2: Urban, 3: Veg
            N_W = Ref_Water.shape[1]
            N_U = Ref_Urban.shape[1]
            N_V = Ref_Vegetation.shape[1]

            Train_R = np.zeros((3, Train_X.shape[1]))
            Train_R[0, :N_W] = 1
            Train_R[1, N_W : N_W + N_U] = 1
            Train_R[2, N_W + N_U :] = 1

            # Train MLP
            # Nodes: [4, 3, 3]
            # Learning Rate Alpha=0.001
            mlp = SimpleMLP(
                input_size=4, hidden_size=3, output_size=3, learning_rate=0.001, seed=1
            )

            print("Training MLP (WashingtonDC)...")
            costs = mlp.train(Train_X, Train_R, epochs=3000)

            # Predict Test Data
            Test_X = np.hstack([Test_Water, Test_Urban, Test_Vegetation])

            N_Wt = Test_Water.shape[1]
            N_Ut = Test_Urban.shape[1]
            N_Vt = Test_Vegetation.shape[1]

            # True Labels (for confusion matrix)
            True_Labels = np.zeros(Test_X.shape[1], dtype=int)
            True_Labels[:N_Wt] = 0  # Water
            True_Labels[N_Wt : N_Wt + N_Ut] = 1  # Urban
            True_Labels[N_Wt + N_Ut :] = 2  # Veg

            # Forward
            Output_Test = mlp.forward(Test_X)

            # Convert to Class Index (Argmax)
            Pred_Labels = np.argmax(Output_Test, axis=0)

            # Confusion Matrix
            # CM = confusion_matrix(True_Labels, Pred_Labels)
            CM = np.zeros((3, 3), dtype=int)
            for t, p in zip(True_Labels, Pred_Labels):
                CM[t, p] += 1
            print("Confusion Matrix:")
            print(CM)

            # Display
            # Figure 1: Original Data + MSE
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(imgs["VisibleBlue"], cmap="gray")
            plt.title("Visible Blue")
            plt.subplot(2, 3, 2)
            plt.imshow(imgs["VisibleGreen"], cmap="gray")
            plt.title("Visible Green")
            plt.subplot(2, 3, 3)
            plt.imshow(imgs["VisibleRed"], cmap="gray")
            plt.title("Visible Red")
            plt.subplot(2, 3, 4)
            plt.imshow(imgs["NearInfraRed"], cmap="gray")
            plt.title("NIR")

            # Mask Composite
            MaskRGB = np.zeros_like(VB)
            MaskCombined = (
                np.dstack(
                    [imgs["UrbanMask"], imgs["VegetationMask"], imgs["WaterMask"]]
                )
                * 255
            )  # R=U, G=V, B=W (Approx)
            # Check MATLAB text: Water(b), urban(r), veg(g)
            # So R=Urban, G=Veg, B=Water
            MaskColor = np.zeros((VB.shape[0], VB.shape[1], 3), dtype=float)
            MaskColor[..., 0] = imgs["UrbanMask"]  # R
            MaskColor[..., 1] = imgs["VegetationMask"]  # G
            MaskColor[..., 2] = imgs["WaterMask"]  # B

            plt.subplot(2, 3, 5)
            plt.imshow(MaskColor)
            plt.title("Masks (U=R, V=G, W=B)")

            out_dir = _os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output"))
            plt.savefig(os.path.join(out_dir, "Figure1337.png"))

            # Figure 2: Confusion Matrix
            plt.figure(figsize=(6, 6))
            plt.imshow(CM, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix (Test Set)")
            plt.colorbar()
            tick_marks = np.arange(3)
            plt.xticks(tick_marks, ["Water", "Urban", "Veg"])
            plt.yticks(tick_marks, ["Water", "Urban", "Veg"])

            thresh = CM.max() / 2.0
            for i in range(CM.shape[0]):
                for j in range(CM.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(CM[i, j], "d"),
                        horizontalalignment="center",
                        color="white" if CM[i, j] > thresh else "black",
                    )

            plt.ylabel("True Class")
            plt.xlabel("Predicted Class")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "Figure1337Bis.png"))

            # MSE
            plt.figure(figsize=(6, 6))
            plt.semilogy(costs)
            plt.xlabel("Iteration")
            plt.title("Train MSE")
            plt.savefig(os.path.join(out_dir, "Figure1339.png"))

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)
