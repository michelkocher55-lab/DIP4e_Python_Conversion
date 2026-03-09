from typing import Any
import numpy as np


def neuralNet4e(input_data: Any, specs: Any):
    """
    Feedforward neural net.

    output = neuralNet4e(input_data, specs)

    Parameters
    ----------
    input_data : dict
        Contains 'X' (Input Patterns), 'R' (Target Patterns), 'Epochs'.
        X: (N, np), N-dim patterns, np patterns.
        R: (nc, np), nc classes.
    specs : dict
        Contains 'Nodes', 'Activation', 'Mode', 'W', 'b', 'Correction'.

    Returns
    -------
    output : dict
        Contains 'W', 'b', 'Class', 'MSE', 'RecogRate', 'A'.
    """

    # Defaults
    if "Epochs" not in input_data:
        input_data["Epochs"] = 1
    if "Mode" in specs and specs["Mode"] == "classify":
        input_data["Epochs"] = 1

    if "Activation" not in specs:
        specs["Activation"] = "sigmoid"
    if "Correction" not in specs:
        specs["Correction"] = 0.1
    if "Mode" not in specs:
        specs["Mode"] = "classify"  # Default per doc

    # Architecture
    X = input_data["X"]
    R = None
    if "R" in input_data:
        R = input_data["R"]

    num_patterns = X.shape[1]  # np
    L = len(specs["Nodes"])  # Number of layers (counting input as 1)

    n = np.zeros(L, dtype=int)
    n[0] = X.shape[0]  # Input dim
    # specs.Nodes includes input layer size in MATLAB example?
    # Doc says: "to specify 4-layer network with 3 inputs... specs.Nodes = [3 5 4 2]."
    # So Nodes[0] should match X rows.
    # MATLAB code: n(1) = size(X,1). n(2:L) = specs.Nodes(2:end).
    # It overwrites n(1).
    n[1:] = specs["Nodes"][1:]  # Copy rest
    # Logic in MATLAB: n(1) = size(X,1) implies specs.Nodes(1) is ignored or expected to match.
    n[0] = X.shape[0]

    # Weights and Biases
    # stored in lists (cell arrays in MATLAB). indices 0..L-1.
    # MATLAB uses 1-based indexing W{k} for layer k (2 to L).
    # We will use W[k], b[k] where k corresponds to layer index 0..L-1.
    # However, W{2} is weights between layer 1 (input) and layer 2.
    # Let's map MATLAB W{k} -> Python W[k-1]? Or keep W size L+1 for clarity?
    # W[k] connects layer k-1 to k?
    # MATLAB: Z{k} = W{k}*A{k-1} + B{k} for k=2:L.
    # So W{k} connects (k-1) -> k.
    # Let's use Python list of size L+1, where index k corresponds to layer k.
    # Index 0 is unused. Index 1 unused for W/b? No, W connects to layer k.
    # W[2] connects layer 1 to 2.
    # This keeps valid indices 2..L matching MATLAB logic.

    W = [None] * (L + 1)
    b = [None] * (L + 1)

    if "W" not in specs:
        # Init W
        for k in range(2, L + 1):
            # MATLAB: W{k} = 0.1 + 0.9*rand(n(k),n(k-1))/1
            # n indices in MATLAB are 1-based. Python 0-based.
            # n[k-1] size of layer k.
            # Wait, `n` vector above: n[0] is layer 1. n[k-1] is layer k.
            rows = n[k - 1]
            cols = n[k - 2]
            W[k] = 0.1 + 0.9 * np.random.rand(rows, cols)
    else:
        # Copy from specs. Assume specs['W'] is list/dict with compatible indexing.
        # If passed as list 0..L-1 from Python caller?
        # User sets specs.W. Let's assume it matches our internal structure or provided as standard list.
        # If standard list [None, None, W2, W3...]?
        W = specs["W"]

    if "b" not in specs:
        # Init b
        for k in range(2, L + 1):
            rows = n[k - 1]
            b[k] = 0.1 + 0.9 * np.random.rand(rows, 1)
    else:
        b = specs["b"]

    output = {}
    output["MSE"] = []

    # Execution
    numLoops = input_data["Epochs"]

    # A, Z, D, Hprime storage
    # A, Z, D, Hprime cell arrays size L+1
    A = [None] * (L + 1)
    Z = [None] * (L + 1)
    D = [None] * (L + 1)  # Delta
    Hprime = [None] * (L + 1)
    d = [None] * (L + 1)  # Bias delta

    for epoch in range(numLoops):
        # Step 1: Init
        A[1] = X

        # Step 2: Forward
        for k in range(2, L + 1):
            # B{k} = repmat(b{k}, 1, np)
            # In numpy broadcasting handles this generally, but strict matrix math:
            # Z = W A + b
            # b is (Nodes, 1). A is (PrevNodes, Batch). W is (Nodes, PrevNodes).
            # b broadcasts to (Nodes, Batch).

            Z[k] = np.dot(W[k], A[k - 1]) + b[k]

            # Activation
            A[k], Hprime[k] = applyH4e(Z[k], specs["Activation"])

        # Step 3: Backprop (Train only)
        if specs["Mode"] == "train":
            if R is None:
                raise ValueError("Mode 'train' requires input_data['R']")

            alpha = specs["Correction"]

            # MSE
            # sum(sum((A{L} - R).^2))/2
            # MATLAB: sum columns, then sum result. (Total sum).
            output["MSE"].append(np.sum((A[L] - R) ** 2) / 2.0)

            # Compute D{L}
            # (A{L} - R) .* Hprime{L}
            D[L] = (A[L] - R) * Hprime[L]

            # Backprop D{k}
            for k in range(L - 1, 1, -1):  # L-1 down to 2
                # D{k} = ((W{k+1})'*D{k+1}).*Hprime{k}
                D[k] = np.dot(W[k + 1].T, D[k + 1]) * Hprime[k]

            # Update weights/biases
            # for k = L:-1:2
            for k in range(L, 1, -1):
                # W{k} = W{k} - alpha*D{k}*(A{k-1})'
                W[k] = W[k] - alpha * np.dot(D[k], A[k - 1].T)

                # d{k} = sum(D{k}, 2)
                # b{k} = b{k} - alpha*d{k}
                d_k = np.sum(D[k], axis=1, keepdims=True)
                b[k] = b[k] - alpha * d_k

            output["X"] = X
            output["R"] = R
            output["W"] = W
            output["b"] = b

    # Output Results
    output["A"] = A  # MATLAB: returns cell array of activations

    # Classify Patterns
    # Find max in A{L} column-wise
    # MATLAB: idx = find(A{L}(:,J)== max(max((A{L}(:,J))))); output.Class(J) = idx(1);
    # Python: argmax uses 0-based index. MATLAB 1-based.
    # But usually classes are 1-based? Or 0-based in input?
    # R is nc-by-np.
    # If using index, it corresponds to class ID 0..nc-1 approx.
    # MATLAB example uses `vec2ind` which likely returns 1..nc.
    # Let's return 0-based index for Python consistency, but note it.
    # Actually, let's look at `R`. If R matches rows, row 0 is class 0/1.
    output["Class"] = np.argmax(A[L], axis=0)

    # RecogRate
    if R is not None:
        truth = np.argmax(R, axis=0)  # Index of 1
        num_errors = np.sum(output["Class"] != truth)
        output["RecogRate"] = ((num_patterns - num_errors) / num_patterns) * 100.0
    else:
        output["RecogRate"] = None

    return output


def applyH4e(Z: Any, mode: Any):
    """applyH4e."""
    if mode == "sigmoid":
        h = 1.0 / (1.0 + np.exp(-Z))
        hprime = h * (1.0 - h)
        return h, hprime
    else:
        raise ValueError("Only sigmoid activation is available.")
