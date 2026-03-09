from typing import Any
import numpy as np


def imPad4e(
    f: Any, R: Any, C: Any, padtype: Any = "zeros", loc: Any = "both", padval: Any = 0
):
    """
    Pads an image by extending its borders.

    g = imPad4e(f, R, C, padtype, loc, padval)

    Parameters:
    f: input image
    R: number of rows to pad
    C: number of columns to pad
    padtype: 'zeros', 'constant', 'replicate'
    loc: 'both', 'pre', 'post'
    padval: value for 'constant' padding (default 0)
    """

    f = np.asarray(f)

    # Determine pad width for rows (axis 0)
    if loc == "both":
        pad_r = (R, R)
        pad_c = (C, C)
    elif loc == "pre":
        pad_r = (R, 0)
        pad_c = (C, 0)
    elif loc == "post":
        pad_r = (0, R)
        pad_c = (0, C)
    else:
        raise ValueError("Unknown loc. Use 'both', 'pre', or 'post'.")

    pad_width = (pad_r, pad_c)

    # Handle multi-channel images (2D or 3D)
    if f.ndim == 3:
        pad_width = pad_width + ((0, 0),)

    if padtype == "zeros":
        g = np.pad(f, pad_width, mode="constant", constant_values=0)
    elif padtype == "constant":
        g = np.pad(f, pad_width, mode="constant", constant_values=padval)
    elif padtype == "replicate":
        g = np.pad(f, pad_width, mode="edge")
    else:
        raise ValueError("Unknown padtype. Use 'zeros', 'constant', or 'replicate'.")

    return g
