from typing import Any
from libDIPUM.wavework import wavework


def wavecut(kind: Any, c: Any, s: Any, level: Any = None):
    """
    Zero coefficients in a wavelet decomposition structure.
    MATLAB wavecut.m wrapper behavior.
    Returns (nc, y).
    """
    if level is None:
        return wavework("cut", kind, c, s)
    return wavework("cut", kind, c, s, level)
