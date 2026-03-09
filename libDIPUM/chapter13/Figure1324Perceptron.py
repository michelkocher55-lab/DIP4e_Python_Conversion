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


if __name__ == "__main__":
    Figure1324Perceptron()
