from typing import Any
from helpers.libdipum.wavework import wavework


def wavepaste(kind: Any, c: Any, s: Any, level: Any, x: Any):
    """
    Paste coefficients into a wavelet decomposition structure.
    MATLAB wavepaste.m wrapper behavior.
    """
    return wavework("paste", kind, c, s, level, x)
