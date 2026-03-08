from libDIPUM.wavework import wavework


def wavecut(kind, c, s, level=None):
    """
    Zero coefficients in a wavelet decomposition structure.
    MATLAB wavecut.m wrapper behavior.
    Returns (nc, y).
    """
    if level is None:
        return wavework('cut', kind, c, s)
    return wavework('cut', kind, c, s, level)
