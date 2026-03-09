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


if __name__ == "__main__":
    Figure1324LMSE()
