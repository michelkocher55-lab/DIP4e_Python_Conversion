import os
import numpy as np


def imratio(f1, f2):
    """
    Compute ratio of bytes in two images/variables.
    """
    return _bytes(f1) / _bytes(f2)


def _bytes(f):
    # If f is a string, treat as filename
    if isinstance(f, str):
        path = f
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return os.path.getsize(path)

    # If f is a dict (struct-like), sum bytes of fields
    if isinstance(f, dict):
        total = 0
        for v in f.values():
            total += _bytes(v)
        return total

    # If numpy array
    if isinstance(f, np.ndarray):
        return f.nbytes

    # If list/tuple, sum items
    if isinstance(f, (list, tuple)):
        return sum(_bytes(v) for v in f)

    # Fallback: estimate by string encoding length
    return len(str(f).encode('utf-8'))
