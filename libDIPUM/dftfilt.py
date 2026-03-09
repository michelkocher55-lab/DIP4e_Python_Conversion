from typing import Any
import numpy as np
from skimage.util import img_as_float


def dftfilt(f: Any, H: Any, pad_method: Any = "constant", class_out: Any = "float"):
    """
    Performs frequency domain filtering.

    Parameters:
    f (ndarray): Input image.
    H (ndarray): Filter transfer function (Uncentered for correct results without manual shift).
    pad_method (str): 'constant' (zeros), 'reflect' (symmetric), 'nearest' (replicate), 'wrap' (circular).
                      Defaults to 'constant' (zeros) if not specified or None.
                      Note: MATLAB 'zeros' -> Python 'constant', cval=0.
    class_out (str): 'float' (default) or 'same'.

    Returns:
    g (ndarray): Filtered image.
    """
    # Note on pad_method mapping:
    # MATLAB 'zeros' -> numpy 'constant', constant_values=0
    # MATLAB 'symmetric' -> numpy 'reflect' (d c b a | a b c d | d c b a)
    # MATLAB 'replicate' -> numpy 'edge' (nearest)
    # MATLAB 'circular' -> numpy 'wrap'

    if pad_method == "zeros" or pad_method == 0 or pad_method is None:
        mode = "constant"
    elif pad_method == "symmetric":
        mode = "reflect"
    elif pad_method == "replicate":
        mode = "edge"
    elif pad_method == "circular":
        mode = "wrap"
    else:
        mode = pad_method  # direct pass if valid numpy mode

    f_float = img_as_float(f)
    M, N = f_float.shape

    P, Q = H.shape

    # Pad f to size of H
    # pad_width = ((0, P-M), (0, Q-N))
    pad_height = P - M
    pad_width_val = Q - N

    if pad_height < 0 or pad_width_val < 0:
        raise ValueError("Filter H must be larger or equal to image f.")

    pad_width = ((0, pad_height), (0, pad_width_val))

    if mode == "constant":
        f_padded = np.pad(f_float, pad_width, mode=mode, constant_values=0)
    else:
        f_padded = np.pad(f_float, pad_width, mode=mode)

    # FFT
    F = np.fft.fft2(f_padded)

    # Apply filter
    # H must be uncentered (DC at corners)
    G = H * F

    # IFFT
    g = np.real(np.fft.ifft2(G))

    # Crop
    g = g[:M, :N]

    if class_out == "same":
        # restore class? usually just return float for simplicity in python
        # or implement conversion based on f type.
        pass

    return g
