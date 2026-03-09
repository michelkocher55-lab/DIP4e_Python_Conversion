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
        costs = mlp.train(Input_Aug, Target_Aug, epochs=500)  # 500 epochs sufficient?
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


if __name__ == "__main__":
    Figure1335()
