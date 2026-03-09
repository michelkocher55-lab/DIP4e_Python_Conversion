from typing import Any
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hpFilterTF4e import hpFilterTF4e
from imageTranslate4e import imageTranslate4e


def notchReject4e(type_str: Any, M: Any, N: Any, param: Any, C: Any):
    """
    Computes a notchreject filter transfer function.

    H = notchReject4e(type_str, M, N, param, C)

    Parameters:
    -----------
    type_str : str
        'ideal', 'gaussian', 'butterworth', or 'rectangle'.
    M : int
        Number of rows.
    N : int
        Number of columns.
    param : float or list
        Filter parameters.
        For 'ideal', 'gaussian': D0.
        For 'butterworth': [D0, n].
        For 'rectangle': [VL, HL, d].
    C : list or str
        Center coordinates [u0, v0] or orientation ('vert', 'horz').

    Returns:
    --------
    H : numpy.ndarray
        M x N transfer function.
    """

    if type_str != "rectangle":
        # Notch is not a rectangle
        u0 = C[0]
        v0 = C[1]

        # Check if indices are valid
        if abs(u0) > M / 2 or abs(v0) > N / 2:
            print("Warning: Your notches may be outside the filter area.")

        # Generate highpass filter (centered at DC)
        # Note: hpFilterTF4e arguments are (type, P, Q, D0, n)
        # We need to unpack param if it's a list for butterworth

        D0 = 0
        n = 1

        if type_str == "butterworth":
            D0 = param[0]
            n = param[1]
        else:
            # Ideal or Gaussian
            if isinstance(param, (list, tuple, np.ndarray)):
                D0 = param[0]
            else:
                D0 = param

        H0 = hpFilterTF4e(type_str, M, N, D0, n)

        # Move it to notch location (u0, v0)
        # imageTranslate4e(f, tx, ty) where tx=RowShift, ty=ColShift
        H1 = imageTranslate4e(H0, u0, v0, mode="white")

        # Move H0 to (-u0, -v0)
        # MATLAB Logic for odd dimensions:
        # if isodd(M), u0 = u0 + 1
        # if isodd(N), v0 = v0 + 1

        u0_sym = u0
        v0_sym = v0

        if M % 2 != 0:
            u0_sym += 1
        if N % 2 != 0:
            v0_sym += 1

        H2 = imageTranslate4e(H0, -u0_sym, -v0_sym, mode="white")

        H = H1 * H2

    elif type_str == "rectangle":
        # Rectangle logic
        VL = param[0]
        HL = param[1]
        d = param[2]

        if C == "vert" and HL % 2 == 0:
            raise ValueError("For a vertical rectangle HL must be odd")
        elif C == "horz" and VL % 2 == 0:
            raise ValueError("For a horizontal rectangle VL must be odd")

        if d > M / 2 - VL or d > N / 2 - HL:
            print("Warning: Your rectangles may be outside the filter area.")

        # Initialize filter
        H0 = np.ones((M, N))

        # Create rectangle at origin (top-left) in MATLAB (1-based) -> (0,0) in Python
        # MATLAB: H0(1:VL, 1:HL) = 0
        # Python: H0[0:VL, 0:HL] = 0
        H0[0 : int(VL), 0 : int(HL)] = 0.0

        # Center of frequency rectangle
        # MATLAB: cu = floor(M/2) + 1; cv = floor(N/2) + 1
        # In Python scaling (0-based indexing), the "center" index is M//2.
        # But imageTranslate works with relative shifts or absolute?
        # imageTranslate4e shifts BY (tx, ty).

        # In MATLAB, the rectangle starts at (1,1).
        # To move (1,1) to (cu, cv), shift is (cu-1, cv-1).
        # In Python, rectangle starts at (0,0).
        # To move (0,0) to Center (M//2, N//2), shift is (M//2, N//2).

        cu = M // 2
        cv = N // 2

        if C == "vert":
            # Center on vertical (u) axis. Displaced by d from center.
            # MATLAB: shift (cu - 1 + d, cv - 1 - (HL-1)/2)
            # Python equivalent:
            # Row Shift: CenterRow + d
            # Col Shift: CenterCol - (HL - 1)/2 -> centering the width HL

            shift_r1 = cu + d
            shift_c1 = cv - (HL - 1) / 2

            H1 = imageTranslate4e(H0, shift_r1, shift_c1, mode="white")

            # Other pair
            # MATLAB: (cu - d - VL, ...)
            # Row Shift: CenterRow - d - VL
            shift_r2 = cu - d - VL
            shift_c2 = shift_c1  # Same col shift

            H2 = imageTranslate4e(H0, shift_r2, shift_c2, mode="white")

        elif C == "horz":
            # Center on horizontal (v) axis.
            # Python:
            # Row Shift: CenterRow - (VL - 1)/2 -> centering height VL
            # Col Shift: CenterCol + d

            shift_r1 = cu - (VL - 1) / 2
            shift_c1 = cv + d

            H1 = imageTranslate4e(H0, shift_r1, shift_c1, mode="white")

            # Other pair
            # Col Shift: CenterCol - d - HL
            shift_c2 = cv - d - HL
            shift_r2 = shift_r1

            H2 = imageTranslate4e(H0, shift_r2, shift_c2, mode="white")

        else:
            raise ValueError("Invalid specification for rectangle orientation")

        H = H1 * H2

    else:
        raise ValueError(f"Unknown filter type: {type_str}")

    return H
