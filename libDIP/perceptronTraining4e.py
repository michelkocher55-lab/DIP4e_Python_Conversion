from typing import Any
import numpy as np


def perceptronTraining4e(
    X: Any, r: Any, alpha: Any = 0.5, nepochs: Any = 100, w0: Any = None
):
    """
    Training of two-class perceptron.

    Parameters
    ----------
    X : numpy.ndarray
        (Dim, NumPatterns). Augmented patterns.
    r : numpy.ndarray
        (NumPatterns,). Class labels (+1, -1).
    alpha : float
        Learning rate.
    nepochs : int or 'max'
        Max epochs.
    w0 : numpy.ndarray, optional
        Initial weights.

    Returns
    -------
    w : numpy.ndarray
        Learned weights.
    actualEpochs : int
        Number of epochs processed.
    """

    num_patterns = X.shape[1]
    dim = X.shape[0]

    if w0 is None:
        np.random.seed(None)  # Shuffle
        w0 = np.random.rand(dim)
    else:
        w0 = np.array(w0).flatten()

    w = w0.copy()
    w_last = w0.copy()

    if nepochs == "max":
        nepochs = 2**31 - 1  # Large int

    actualEpochs = 0

    # Ensure r is flat
    r = np.array(r).flatten()

    for I in range(nepochs):
        actualEpochs += 1
        convergence = True

        for J in range(num_patterns):
            # Check prediction
            # sign(w' * X(:, J))
            activation = np.dot(w, X[:, J])
            pred_sign = np.sign(activation)
            if pred_sign == 0:
                pred_sign = 1  # ? MATLAB sign(0) = 0.
                # MATLAB code: if sign(...) ~= r(J).
                # r(J) is +1/-1. So if 0, it's an error.
                # If pred_sign is 0, it != r(J).

            # Use strict inequality logic or sign matching
            if pred_sign != r[J]:
                # Update
                # w = w_last + r(J)*alpha*X(:,J)
                w = w_last + r[J] * alpha * X[:, J]
                convergence = False
                w_last = w.copy()

        if convergence:
            break

    return w, actualEpochs
