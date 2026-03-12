from typing import Any
import numpy as np
from scipy.signal import convolve2d

try:
    from intScaling4e import intScaling4e
    from imPad4e import imPad4e
except ImportError:
    from DIP4eFigures.intScaling4e import intScaling4e
    from DIP4eFigures.imPad4e import imPad4e


def twodConv4e(f: Any, w: Any, param: Any = "s"):
    """
    Performs 2D convolution matches MATLAB behavior.

    Parameters:
        f: Input image.
        w: Kernel.
        param: 's' (scale input to 0-1 float), 'ns' (no scaling, just double).

    Returns:
        g: Convolved image (same size).
    """
    # Create copies to avoid side effects if any
    if param == "s":
        f = intScaling4e(f)
    elif param == "ns":
        f = np.asarray(f).astype(float)
    else:
        # Default 's'? MATLAB code checks (nargin==2 || param=='s') vs param=='ns'.
        # Assuming 's' is default if logic falls through?
        # MATLAB Line 28: if nargin==2 || ... param='s' logic runs.
        f = intScaling4e(f)

    w = np.asarray(w).astype(float)

    # MATLAB logic: rep padding to avoid zero padding of conv2.
    # [M,N] = size(f)
    # [WR,WC] = size(w)
    # rowBorder = ceil((WR - 1)/2)
    # colBorder = ceil((WC - 1)/2)
    # f = imPad4e(f, rowBorder, colBorder, 'replicate', 'both')
    # g = conv2(f, w, 'same')
    # strip buffer

    M, N = f.shape[0], f.shape[1]  # Handle multichannel? function assumes 2D usually?
    # MATLAB conv2 is 2D.
    # If F is multichannel, twodConv4e probably fails or iterates?
    # MATLAB code doesn't explicitly loop channels, usually implies 2D.

    WR, WC = w.shape
    rowBorder = int(np.ceil((WR - 1) / 2))
    colBorder = int(np.ceil((WC - 1) / 2))

    # Pad
    f_padded = imPad4e(f, rowBorder, colBorder, "replicate", "both")

    # Convolve
    # scipy.signal.convolve2d(in1, in2, mode='same', boundary='fill', fillvalue=0)
    # boundary='fill', fillvalue=0 matches MATLAB conv2 zero padding assumption on input.
    # Since we padded F with replicate, conv2's zero padding affects the 'replicate' border, not original F.
    # And we crop 'same' size as f_padded.
    # Then we crop again manually.

    g_full = convolve2d(f_padded, w, mode="same", boundary="fill", fillvalue=0)

    # Strip padding
    # g = g(rowBorder + 1:M + rowBorder, colBorder + 1:N + colBorder);
    # 0-indexed: rowBorder : rowBorder + M

    g = g_full[rowBorder : rowBorder + M, colBorder : colBorder + N]

    return g
