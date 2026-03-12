from typing import Any
import numpy as np


def GPsnr(x: Any, y: Any):
    """
    Signal-to-noise ratio:
    v = 20*log10(norm(x(:)) / norm(x(:)-y(:)))

    Parameters
    ----------
    x : array_like
        Original clean signal (reference).
    y : array_like
        Denoised signal.

    Returns
    -------
    float
        SNR value in dB.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    num = np.linalg.norm(x.ravel())
    den = np.linalg.norm((x - y).ravel())

    if den == 0:
        return np.inf

    return 20.0 * np.log10(num / den)
