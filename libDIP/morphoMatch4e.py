from typing import Any
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def morphoMatch4e(I: Any, B: Any, padval: Any = 0, mode: Any = "same"):
    """
    Finds matches of a structuring element B in binary image I.

    S = morphoMatch4e(I, B, padval=0, mode='same')

    Parameters
    ----------
    I : numpy.ndarray
        Binary image (0s and 1s).
    B : numpy.ndarray
        Structuring element (m-by-n, odd dims).
        values: 0, 1, or other (Don't Care).
    padval : int/float
        Padding value (0 or 1). Default 0.
    mode : str
        'same' (return size of I) or 'full' (return size of padded I).

    Returns
    -------
    S : numpy.ndarray
        Match score map (0: No match, 0.5: Partial, 1: Perfect).
    """

    I = np.array(I, dtype=float)
    B = np.array(B)

    # Validation
    if not np.all(np.isin(I, [0, 1])):
        # Try to binarize or warn? MATLAB code errors if not exactly 0/1 after float conversion?
        # MATLAB: "I(I>0)=1". Line 63.
        I = (I > 0).astype(float)

    M, N = I.shape
    m, n = B.shape

    # Padding I
    # MATLAB: padarray(I, [m, n], padval) -> pads m rows top/bot, n cols left/right.
    Ip = np.pad(I, ((m, m), (n, n)), mode="constant", constant_values=padval)

    # Perform Sliding Window Matching on Ip
    # We want output S to match Ip size.
    # To get output size match Ip, we need to pad Ip further for the convolution/sliding?
    # Center of B aligned with pixel (r, c) of Ip.
    # We need padding of (m-1)//2, (n-1)//2.

    pad_h = (m - 1) // 2
    pad_w = (n - 1) // 2

    # Pad Ip with 0s (implicit colfilt behavior)
    Ip_padded = np.pad(
        Ip, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
    )

    # Extract windows
    # Window shape: (m, n)
    windows = sliding_window_view(Ip_padded, window_shape=(m, n))

    # sliding_window_view returns shape (Rows, Cols, m, n)
    # Output dims should match Ip.
    # Ip_padded is (Mp + 2pad_h, Np + 2pad_w).
    # Output Rows = (Mp + 2pad_h) - m + 1 = Mp + (m-1) - m + 1 = Mp.
    # So `windows` shape is (Mp, Np, m, n). Correct.

    # Flatten windows and B for comparison
    # But B has 'don't care'.

    B_flat = B  # Keep shape to broadcast? No, B is (m,n).

    # Identify Don't Care locations
    # Don't Care: Not 0 AND Not 1.
    is_care = (B == 0) | (B == 1)
    is_dont_care = ~is_care

    # Helper to check matches
    # Match condition:
    # 1. Pixel match (A == B)
    # 2. OR Don't Care (B is dont care) -> Force True.

    # windows is (H, W, m, n). B is (m, n).
    # Broadcast B.

    # Exact matches (Values agree)
    values_agree = windows == B

    # Condition: values_agree OR is_dont_care
    # Note: is_dont_care broadcasts over H, W.
    matches = values_agree | is_dont_care

    # Count matches per window
    # Sum over last two axes (m, n)
    match_counts = np.sum(matches, axis=(-2, -1))

    total_pixels = m * n

    S = np.zeros_like(match_counts, dtype=float)

    # Perfect Match
    S[match_counts == total_pixels] = 1.0

    # Partial Match
    # C > 0 and C < total
    S[(match_counts > 0) & (match_counts < total_pixels)] = 0.5

    # Crop S if mode is 'same'
    # S corresponds to Ip.
    # Ip size: (M + 2m, N + 2n)
    # Target 'same' crop: Recover I region.
    # Matches MATLAB: `S = S(m+1:M+m, n+1:N+n)` (1-based)
    # padding was m top, m bot.
    # I is in the middle.
    # Indices: m to m+M.

    if mode == "same":
        S = S[m : m + M, n : n + N]

    return S
