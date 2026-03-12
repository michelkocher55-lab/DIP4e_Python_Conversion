from typing import Any
import numpy as np
from scipy import signal


def sigm(x: Any):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def cnnff(net: Any, x: Any):
    """
    CNN Feed Forward.

    net = cnnff(net, x)

    Parameters
    ----------
    net : dict
        CNN network structure.
    x : numpy.ndarray
        Input data (Height, Width, BatchSize).

    Returns
    -------
    net : dict
        Network with updated activations 'a', feature vector 'fv', and output 'o'.
    """

    n = len(net["layers"])

    # Layer 0: Input
    # MATLAB: net.layers{1}.a{1} = x;
    # Python: layers[0]['a'] = [x]
    # Ensure structure exists
    if "a" not in net["layers"][0]:
        net["layers"][0]["a"] = [None]
    net["layers"][0]["a"][0] = x

    inputmaps = 1

    # Loop layers 1 to n-1 (MATLAB 2 to n)
    for l in range(1, n):
        layer = net["layers"][l]
        prev_layer = net["layers"][l - 1]

        if layer["type"] == "c":
            # Initialize 'a'
            # In MATLAB, 'outputmaps' is stored in layer.
            # We assume logical consistency: layer['a'] size = outputmaps.
            # But during FF we overwrite/create 'a'.
            outputmaps = layer["outputmaps"]
            layer["a"] = [None] * outputmaps

            for j in range(outputmaps):
                # z = zeros(...)
                # Size: PrevSize - KernelSize + 1
                # prev_layer['a'][0].shape is (H, W, Batch)

                # Sum over input maps
                z = None

                for i in range(inputmaps):
                    # convn(prev.a{i}, kernel{i}{j}, 'valid')
                    input_map = prev_layer["a"][i]
                    kernel = layer["k"][i][j]

                    # Scipy Convolve
                    # input_map: (H, W, Batch)
                    # kernel: (Kh, Kw) -> Broadcast to (Kh, Kw, 1) to match batch dim
                    res = signal.convolve(
                        input_map, kernel[..., np.newaxis], mode="valid"
                    )

                    if z is None:
                        z = res
                    else:
                        z = z + res

                # Add bias and sigm
                # b{j} is scalar or 1D? Usually scalar per map.
                # MATLAB: z + net.layers{l}.b{j}
                layer["a"][j] = sigm(z + layer["b"][j])

            inputmaps = outputmaps

        elif layer["type"] == "s":
            # Downsample (Average Pooling)
            scale = layer["scale"]
            layer["a"] = [None] * inputmaps

            # Kernel for Avg Pooling: ones(scale)/scale^2
            pool_kernel = np.ones((scale, scale)) / (scale**2)

            for j in range(inputmaps):
                input_map = prev_layer["a"][j]

                # Convolve valid
                z = signal.convolve(
                    input_map, pool_kernel[..., np.newaxis], mode="valid"
                )

                # Downsample slices: z(1:scale:end, 1:scale:end, :)
                # Actually MATLAB starts at 1. If Python 0-indexed?
                # MATLAB 1:scale:end means indices 1, 1+scale, ... (1-based)
                # equal to indices 0, scale, ... (0-based)
                # Wait. No.
                # MATLAB 1-based: Index 1 is the first pixel.
                # Python 0-based: Index 0 is the first pixel.
                # So slice [::scale, ::scale] matches.

                layer["a"][j] = z[::scale, ::scale, :]

    # Feature Vector
    # Concatenate all end layer feature maps
    # MATLAB: reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))
    # Concatenate vertically [fv; new]

    last_layer = net["layers"][-1]
    fv_list = []

    for j in range(len(last_layer["a"])):
        feature_map = last_layer["a"][j]  # (H, W, Batch)
        H, W, B = feature_map.shape
        # Reshape to (H*W, B)
        # Default order C. MATLAB is F?
        # Check alignment with weights ffW.
        # If weights trained with MATLAB, mismatch is fatal.
        # But if training is consistent (all Python), C is fine.
        # Let's stick to default C for now.
        flattened = feature_map.reshape(H * W, B)
        fv_list.append(flattened)

    if fv_list:
        net["fv"] = np.concatenate(fv_list, axis=0)
    else:
        net["fv"] = np.empty((0, 0))  # Should not happen

    # Feedforward Output
    # net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)))
    # ffW: (OutDim, FeatDim). fv: (FeatDim, Batch).
    # result: (OutDim, Batch).
    # ffb: (OutDim, 1) broadcasts automatically in NumPy.

    net["o"] = sigm(np.dot(net["ffW"], net["fv"]) + net["ffb"])

    return net
