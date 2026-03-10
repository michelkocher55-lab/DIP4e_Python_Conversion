from typing import Any
import numpy as np


def cnnsetup(net: Any, x: Any, y: Any):
    """
    Initialize CNN weights and biases.

    net = cnnsetup(net, x, y)

    Parameters
    ----------
    net : dict
        CNN structure with 'layers' list.
    x : numpy.ndarray
        Input data (Height, Width, Batch). Used to determine map size.
    y : numpy.ndarray
        Target labels (Classes, Batch). Used to determine output size.

    Returns
    -------
    net : dict
        Initialized network.
    """

    inputmaps = 1
    # mapsize: (H, W). Assuming x is (H, W, Batch) or (H, W) or (H, W, C)?
    # MATLAB: size(squeeze(x(:, :, 1)))
    # If x is 3D (H,W,B), x(:,:,1) is (H,W).
    if x.ndim == 3:
        mapsize = np.array(x.shape[:2])
    elif x.ndim == 2:
        mapsize = np.array(x.shape)
    else:
        # Assuming last dim is batch
        mapsize = np.array(x.shape[:-1])

    for l, layer in enumerate(net["layers"]):
        if layer["type"] == "s":
            scale = layer["scale"]
            # Divide mapsize
            # mapsize = mapsize / scale
            # Check integer division
            if np.any(mapsize % scale != 0):
                raise ValueError(
                    f"Layer {l} (Subsampling): Map size {mapsize} not divisible by scale {scale}"
                )

            mapsize = mapsize // scale

            # Init biases to 0? MATLAB: net.layers{l}.b{j} = 0;
            # For 's' layer, usually fixed, but code sets b=0.
            # In 's' layer usually 'b' is learned if trainable, but here code just sets to 0.
            layer["b"] = [0.0] * inputmaps

            # No kernels for 's' in this toolbox (fixed avg pool)?

        elif layer["type"] == "c":
            kernelsize = layer["kernelsize"]
            outputmaps = layer["outputmaps"]

            # Update mapsize: H - K + 1
            mapsize = mapsize - kernelsize + 1

            # Init Fan-In / Fan-Out for Xavier
            fan_out = outputmaps * (kernelsize**2)
            fan_in = inputmaps * (kernelsize**2)

            limit = np.sqrt(6 / (fan_in + fan_out))

            layer["k"] = [[None for _ in range(outputmaps)] for _ in range(inputmaps)]
            layer["b"] = [0.0] * outputmaps

            for j in range(outputmaps):
                for i in range(inputmaps):
                    # (rand - 0.5) * 2 * limit -> Uniform(-limit, limit)
                    # np.random.uniform(-limit, limit, size)

                    k_shape = (kernelsize, kernelsize)
                    layer["k"][i][j] = np.random.uniform(-limit, limit, k_shape)

                layer["b"][j] = 0.0

            inputmaps = outputmaps

    # Fully Connected Init
    # fvnum = prod(mapsize) * inputmaps
    fvnum = np.prod(mapsize) * inputmaps

    # onum = size(y, 1) -> Classes
    onum = y.shape[0]

    net["ffb"] = np.zeros((onum, 1))

    # ffW
    fan_in_ff = fvnum
    fan_out_ff = onum
    limit_ff = np.sqrt(6 / (fan_in_ff + fan_out_ff))

    net["ffW"] = np.random.uniform(-limit_ff, limit_ff, (onum, fvnum))

    return net
