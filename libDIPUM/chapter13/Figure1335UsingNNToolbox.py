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
                delta1 = np.dot(self.W2.T, delta2) * (self.a1 * (1 - self.a1))

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


if __name__ == "__main__":
    Figure1335UsingNNToolbox()
