from typing import Any
import numpy as np


def imPad4e(
    f: Any, R: Any, C: Any, padtype: Any = "zeros", loc: Any = "both", padval: Any = 0
):
    """
    Pads an image by extending its borders.

    Parameters:
        f: Input image.
        R, C: Number of rows/cols to pad.
        padtype: 'zeros', 'constant', 'replicate'.
        loc: 'both', 'pre', 'post'.
        padval: Value for 'constant' padding (default 0).

    Returns:
        g: Padded image.
    """
    f = np.asarray(f)

    # Scipy/Numpy pad options
    # 'zeros' -> mode='constant', constant_values=0
    # 'constant' -> mode='constant', constant_values=padval
    # 'replicate' -> mode='edge' (replicates edge values)

    np_mode = "constant"
    kwargs = {}

    if padtype == "zeros":
        np_mode = "constant"
        kwargs["constant_values"] = 0
    elif padtype == "constant":
        np_mode = "constant"
        kwargs["constant_values"] = padval
    elif padtype == "replicate":
        np_mode = "edge"
    else:
        raise ValueError(f"Unknown padtype: {padtype}")

    # Location logic
    # dim 0 (rows), dim 1 (cols)
    pad_width = []

    if loc == "both":
        pad_width = ((R, R), (C, C))
    elif loc == "pre":
        pad_width = ((R, 0), (C, 0))
    elif loc == "post":
        pad_width = ((0, R), (0, C))
    else:
        # Default logic in MATLAB was 'both' if unspecified.
        # But if 'loc' is garbage, error? MATLAB didn't check garbage explicitly but switch used default case?
        # No, MATLAB switch logic checks specific cases.
        raise ValueError(f"Unknown loc: {loc}")

    # Handle multichannel?
    if f.ndim == 3:
        # Pad 3rd dim with 0
        pad_width = list(pad_width)
        pad_width.append((0, 0))
        pad_width = tuple(pad_width)

    g = np.pad(f, pad_width, mode=np_mode, **kwargs)

    return g
