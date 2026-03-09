from typing import Any
import numpy as np


def padarray(a: Any, padsize: Any, padval: Any = 0, direction: Any = "both"):
    """
    Python equivalent of MATLAB padarray for numeric/logical arrays.

    padarray(A, padsize)
    padarray(A, padsize, padval)
    padarray(A, padsize, padval_or_method, direction)

    Supported methods: 'constant', 'circular', 'replicate', 'symmetric'
    Directions: 'pre', 'post', 'both'
    """
    a = np.asarray(a)

    # Normalize padsize
    padsize = np.atleast_1d(padsize).astype(int)
    if padsize.size < a.ndim:
        padsize = np.pad(padsize, (0, a.ndim - padsize.size), mode="constant")
    if padsize.size > a.ndim:
        padsize = padsize[: a.ndim]

    method = "constant"
    if isinstance(padval, str):
        method = padval
        padval = 0

    # Direction handling
    direction = direction.lower()
    if direction == "pre":
        pad_width = [(p, 0) for p in padsize]
    elif direction == "post":
        pad_width = [(0, p) for p in padsize]
    else:
        pad_width = [(p, p) for p in padsize]

    method = method.lower()
    if method == "constant":
        return np.pad(a, pad_width, mode="constant", constant_values=padval)
    if method == "circular":
        return np.pad(a, pad_width, mode="wrap")
    if method == "replicate":
        return np.pad(a, pad_width, mode="edge")
    if method == "symmetric":
        return np.pad(a, pad_width, mode="symmetric")

    raise ValueError(f"Unsupported padding method: {method}")
