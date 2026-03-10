from typing import Any
from helpers.wavework import wavework


def wavecopy(kind: Any, c: Any, s: Any, level: Any = None):
    """
    Fetch coefficients from a wavelet decomposition structure.
    MATLAB wavecopy.m wrapper behavior.
    """
    if level is None:
        return wavework("copy", kind, c, s)
    return wavework("copy", kind, c, s, level)
