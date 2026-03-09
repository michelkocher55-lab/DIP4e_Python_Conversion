from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from lib.cnnsetup import cnnsetup
from lib.cnntrain import cnntrain
from lib.cnntest import cnntest
from lib.cnnff import cnnff


def convNet4e(input_data: Any, specs: Any):
    """
    Deep convolutional network wrapper.

    output = convNet4e(input_data, specs)

    Parameters
    ----------
    input_data : dict
        Contains 'I' (Images), 'R' (Labels), 'cnn' (Trained Net), etc.
        Keys:
            'I': (H, W, M) or (H, W, C, M)
            'R': (Classes, M) - One-hot labels
            'cnn': Trained network (for classify mode)
            'BatchSize': int
            'Epochs': int
            'Alpha': float
            'DisplayData': list/string ('MSEPlot', 'ErrorRate')

    specs : dict
        Contains architecture specs and mode.
        Keys:
            'FM': list of feature map counts per layer pair.
            'KS': list of kernel sizes.
            'SP': list/scalar of subsampling pooling sizes.
            'Mode': 'train' or 'classify'.

    Returns
    -------
    output : dict
        Results containing 'cnn' (trained), 'TrainError', 'ClassLabels', etc.
    """

    # Defaults
    if "Alpha" not in input_data:
        input_data["Alpha"] = 1.0

    if "SP" not in specs:
        specs["SP"] = [2] * len(specs["FM"])  # Default 2 if not provided?
    elif np.isscalar(specs["SP"]):
        specs["SP"] = [specs["SP"]] * len(specs["FM"])

    if "Mode" not in specs:
        specs["Mode"] = "train"

    if specs["Mode"] == "train" and "BatchSize" not in input_data:
        # Default batchsize = number of samples? Or classes?
        # MATLAB: size(input.R, 1) which is number of Classes? odd default.
        # Let's assume reasonable default or warn.
        # Check if R exists
        if "R" in input_data:
            input_data["BatchSize"] = input_data["R"].shape[0]  # Usually small.
        else:
            input_data["BatchSize"] = 10  # Fallback

    output = {}

    # --- Classify Mode ---
    if specs["Mode"] == "classify":
        if "cnn" not in input_data:
            raise ValueError("Mode 'classify' requires 'cnn' in input_data.")

        trained_net = input_data["cnn"]

        if "R" in input_data:
            # Have ground truth -> Run Test
            err_rate, _, _, est = cnntest(trained_net, input_data["I"], input_data["R"])
            output["TestError"] = err_rate * 100.0
            output["ClassLabels"] = est
            output["RLabels"] = np.argmax(input_data["R"], axis=0)
        else:
            # Just predict -> Run FF
            net = cnnff(trained_net, input_data["I"])
            est = np.argmax(net["o"], axis=0)  # Indices
            output["ClassLabels"] = est

        return output

    # --- Train Mode ---
    else:
        # Setup Architecture
        # Loop over FM (Feature Maps)
        num_layers = len(specs["FM"])

        cnn = {}
        cnn["layers"] = []

        # Layer 0: Input 'i'
        cnn["layers"].append({"type": "i"})

        # Build layers pairs (Conv, Sub)
        for i in range(num_layers):
            # Conv Layer
            fm = specs["FM"][i]
            ks = specs["KS"][i]
            cnn["layers"].append({"type": "c", "outputmaps": fm, "kernelsize": ks})

            # Sub Layer
            sp = specs["SP"][i]
            cnn["layers"].append({"type": "s", "scale": sp})

        # Initialize
        cnn = cnnsetup(cnn, input_data["I"], input_data["R"])

        # Train Options
        opts = {
            "alpha": input_data["Alpha"],
            "batchsize": input_data["BatchSize"],
            "numepochs": input_data["Epochs"],
        }

        # Train
        cnn = cnntrain(cnn, input_data["I"], input_data["R"], opts)

        # Output
        output["cnn"] = cnn

        # Evaluate Training Error
        err_rate, _, _, est = cnntest(cnn, input_data["I"], input_data["R"])

        output["TrainError"] = err_rate * 100.0
        output["ClassLabels"] = est
        output["RLabels"] = np.argmax(input_data["R"], axis=0)
        output["TrainMSE"] = cnn["rL"]

        # Displays
        display_opts = input_data.get("DisplayData", [])
        if isinstance(display_opts, str):
            display_opts = [display_opts]

        if "ErrorRate" in display_opts:
            print(f"Error rate (training) = {output['TrainError']:.2f}%")

        if "MSEPlot" in display_opts:
            plt.figure(num="Batch-wise mean squared error during training")
            plt.plot(cnn["rL"], linewidth=2)
            plt.xlabel("Batches")
            plt.ylabel("Mean squared error")
            plt.grid(True)
            # plt.show() # Blocking? Maybe just save/create artifact?
            # User usually expects show if interactive.
            # I won't block here in library code, but caller might.

    return output
