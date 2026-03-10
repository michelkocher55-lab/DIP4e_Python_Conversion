from typing import Any
import numpy as np
import time
from helpers.cnnff import cnnff
from helpers.cnnbp import cnnbp
from helpers.cnnapplygrads import cnnapplygrads


def cnntrain(net: Any, x: Any, y: Any, opts: Any):
    """
    Train CNN using Stochastic Gradient Descent.

    net = cnntrain(net, x, y, opts)

    Parameters
    ----------
    net : dict
        CNN network.
    x : numpy.ndarray
        Input data (Height, Width, Samples).
    y : numpy.ndarray
        Target labels (Classes, Samples).
    opts : dict
        Options:
            'batchsize': int
            'numepochs': int
            'alpha': float (learning rate)

    Returns
    -------
    net : dict
        Trained network. Contains 'rL' (Running Loss history).
    """

    m = x.shape[2]  # Number of samples
    batchsize = opts["batchsize"]
    numepochs = opts["numepochs"]

    numbatches = m / batchsize
    if numbatches % 1 != 0:
        raise ValueError("numbatches not integer. m must be divisible by batchsize.")
    numbatches = int(numbatches)

    net["rL"] = []

    for i in range(numepochs):
        print(f"Epoch {i + 1}/{numepochs}")
        start_time = time.time()

        # Shuffle
        kk = np.random.permutation(m)

        for l in range(numbatches):
            # Batch Indices
            start_idx = l * batchsize
            end_idx = (l + 1) * batchsize
            batch_indices = kk[start_idx:end_idx]

            # Slice Data
            # x is (H, W, m)
            batch_x = x[:, :, batch_indices]
            # y is (Classes, m)
            batch_y = y[:, batch_indices]

            # Feed Forward
            net = cnnff(net, batch_x)

            # Back Propagation
            net = cnnbp(net, batch_y)

            # Update Parameters
            net = cnnapplygrads(net, opts)

            # Running Loss Update
            if not net["rL"]:
                net["rL"].append(net["L"])

            # EMA of Loss
            # net.rL(end + 1) = .9 * net.rL(end) + .1 * net.L;
            new_rL = 0.9 * net["rL"][-1] + 0.1 * net["L"]
            net["rL"].append(new_rL)

        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.4f}s")

    return net
