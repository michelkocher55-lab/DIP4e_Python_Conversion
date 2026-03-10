from typing import Any
import numpy as np


def recnotch(notch: Any, mode: Any, M: Any, N: Any, W: Any = 3, S: Any = 1):
    """
    Generates axes notch filter transfer functions.

    H = recnotch(notch, mode, M, N, W, S)

    Parameters:
    notch: 'reject' or 'pass'
    mode: 'vertical', 'horizontal', or 'both'
    M, N: Size of the transfer function
    W: Width of the rectangles (must be odd, default 3)
    S: Start of the rectangles from center (default 1).
       Can be scalar or [SV, SH] for 'both' mode.

    Returns:
    H: M-by-N notch filter transfer function (uncentered).
    """

    # Defaults handled by args but MATLAB logic handles varargin.
    # Python explicit args cover it using defaults.

    if W % 2 == 0:
        raise ValueError("W must be odd.")

    # Setup AV, AH, SV, SH
    # 0 for reject zone, 1 for pass zone (in reject filter logic)

    if mode == "vertical":
        # Reject vertical only
        AV = 0
        AH = 1
        SV = S
        SH = 1  # Not used for rejection
    elif mode == "horizontal":
        # Reject horizontal only
        AV = 1
        AH = 0
        SH = S
        SV = 1
    elif mode == "both":
        # Reject both
        AV = 0
        AH = 0

        # S can be scalar or [SV, SH]
        # In Python if S is list/tuple:
        if np.ndim(S) == 0:
            SV = S
            SH = S
        elif len(S) == 1:
            SV = S[0]
            SH = S[0]
        else:
            SV = S[0]
            SH = S[1]
    else:
        raise ValueError("Unknown mode. Use 'vertical', 'horizontal', or 'both'.")

    # Generate reject filter
    H = rectangleReject(M, N, W, SV, SH, AV, AH)

    # Uncenter
    H = np.fft.ifftshift(H)

    # Pass mode
    if notch == "pass":
        H = 1 - H

    return H


def rectangleReject(M: Any, N: Any, W: Any, SV: Any, SH: Any, AV: Any, AH: Any):
    """rectangleReject."""
    H = np.ones((M, N), dtype=float)

    # Center (0-based)
    # MATLAB: floor(M/2) + 1.
    # Python: M // 2.
    UC = M // 2
    VC = N // 2

    # Width limits
    # MATLAB: WL = (W - 1) / 2
    WL = int((W - 1) // 2)

    # Python Slicing: [start:end] excludes end.

    # Left, horizontal rectangle
    # MATLAB: H(UC-WL : UC+WL, 1 : VC-SH) = AH
    # Python Rows: UC-WL to UC+WL+1
    # Python Cols: 0 to VC-SH
    # Note: VC is center index. VC-SH is index to stop at?
    # MATLAB 1:VC-SH means up to column index VC-SH (inclusive).
    # If VC=100, SH=1. Ends at 99.
    # Python 0:VC-SH (excludes VC-SH). 0..VC-SH-1.
    # This matches counting.
    H[UC - WL : UC + WL + 1, 0 : int(VC - SH)] = AH

    # Right, horizontal rectangle
    # MATLAB: H(UC-WL : UC+WL, VC+SH : N) = AH
    # Python Cols: VC+SH to N
    H[UC - WL : UC + WL + 1, int(VC + SH) : N] = AH

    # Top vertical rectangle
    # MATLAB: H(1 : UC-SV, VC-WL : VC+WL) = AV
    # Python Rows: 0 to UC-SV
    # Python Cols: VC-WL to VC+WL+1
    H[0 : int(UC - SV), VC - WL : VC + WL + 1] = AV

    # Bottom vertical rectangle
    # MATLAB: H(UC+SV : M, VC-WL : VC+WL) = AV
    # Python Rows: UC+SV to M
    H[int(UC + SV) : M, VC - WL : VC + WL + 1] = AV

    return H
