from typing import Any
import numpy as np
from libDIPUM.cnnff import cnnff


def cnntest(net: Any, x: Any, y: Any):
    """
    Test CNN performance.

    err_rate, net, bad, est = cnntest(net, x, y)

    Parameters
    ----------
    net : dict
        CNN network.
    x : numpy.ndarray
        Input data.
    y : numpy.ndarray
        Target labels (Classes x Batch). One-hot or probabilities.

    Returns
    -------
    err_rate : float
        Classification error rate.
    net : dict
        Network after feedforward.
    bad : numpy.ndarray
        Indices of misclassified samples.
    est : numpy.ndarray
        Estimated class labels (indices).
    """

    # 1. Feedforward
    net = cnnff(net, x)

    # 2. Predictions
    # net.o is (Classes, Batch). Max over classes (axis 0).
    # MATLAB max returns values and indices. We want indices.
    est = np.argmax(net["o"], axis=0)

    # 3. Ground Truth
    # y is (Classes, Batch).
    tru = np.argmax(y, axis=0)

    # 4. Find Errors
    # Indices where est != tru
    bad = np.where(est != tru)[0]

    # 5. Error Rate
    num_bad = len(bad)
    total = y.shape[1]

    err_rate = num_bad / float(total)

    return err_rate, net, bad, est
