from libDIPUM.wavework import wavework


def wavecopy(kind, c, s, level=None):
    """
    Fetch coefficients from a wavelet decomposition structure.
    MATLAB wavecopy.m wrapper behavior.
    """
    if level is None:
        return wavework('copy', kind, c, s)
    return wavework('copy', kind, c, s, level)
