from libDIPUM.wavework import wavework


def wavepaste(kind, c, s, level, x):
    """
    Paste coefficients into a wavelet decomposition structure.
    MATLAB wavepaste.m wrapper behavior.
    """
    return wavework('paste', kind, c, s, level, x)
