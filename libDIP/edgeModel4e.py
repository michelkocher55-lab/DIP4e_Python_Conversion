from typing import Any
import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from imageRotate4e import imageRotate4e
except ImportError:
    pass


def edgeModel4e(
    edge_type: Any,
    m: Any,
    n: Any,
    i_low: Any,
    i_high: Any,
    width: Any = None,
    angle: Any = 0.0,
):
    """
    Generates image of an edge of a specified type.

    Parameters:
    -----------
    edge_type : str
        'step', 'ramp', or 'roof'.
    m, n : int
        Size of the output image. Must be even.
    i_low : float
        Intensity on the left (or base).
    i_high : float
        Intensity on the right (or peak).
    width : int, optional
        Width of the edge (ramp or roof). Defaults to round(n/4) (+1 for roof).
    angle : float, optional
        Rotation angle in degrees. Default 0.

    Returns:
    --------
    f : numpy.ndarray
        Generated edge image.
    """
    if m % 2 != 0 or n % 2 != 0:
        raise ValueError("m and n must be even integers.")
    if i_low >= i_high:
        raise ValueError("i_low must be less than i_high.")

    # Defaults
    if width is None:
        if edge_type == "roof":
            width = round(n / 4) + 1
        else:
            width = round(n / 4)

    # Double size
    m_orig, n_orig = m, n
    m = 2 * m
    n = 2 * n

    # Create profile
    midpoint = (
        int(np.floor(n / 2)) + 1
    )  # 1-based index in MATLAB logic. Python index: midpoint-1
    # Let's stick to Python 0-based indexing for array access but keep logic similar

    # Python midpoint index
    mid_idx = int(n / 2)  # n is 2*N, even. n/2 is integer. e.g. N=4 -> n=8 -> mid=4.
    # MATLAB: midpoint = N/2 + 1. e.g. 8/2 + 1 = 5.
    # Python indices: 0..7. MATLAB indices: 1..8.
    # MATLAB 5 is Python 4.
    # So mid_idx = n // 2 is correct mapping.

    profile = np.zeros(n)

    # Initial step profile
    # profile(1:midpoint-1) = iLow
    # profile(midpoint:N) = iHigh
    profile[:mid_idx] = i_low
    profile[mid_idx:] = i_high

    # First and last point calculation
    if edge_type == "roof" and (width % 2 == 0):
        width += 1

    first_point = mid_idx - int(np.floor(width / 2))  # Python index
    # MATLAB: firstpoint = midpoint - floor(width/2)
    # If mid=5 (Py4), w=3. floor(1.5)=1. 5-1=4 (Py3).
    # Python: 4 - 1 = 3. Correct.

    last_point = first_point + width - 1

    if edge_type == "step":
        pass

    elif edge_type == "ramp":
        if width == 1:
            pass
        else:
            delta = (i_high - i_low) / (width - 1)
            profile[first_point] = i_low
            profile[last_point] = i_high

            # loop firstpoint+1 to lastpoint-1
            # Python range(first_point + 1, last_point) (exclusive end) matches firstpoint+1:lastpoint-1 in effect?
            # Wait, MATLAB: for k = firstpoint+1 : firstpoint+width-2
            # lastpoint is firstpoint + width - 1.
            # So lastpoint - 1 is firstpoint + width - 2.
            # Yes.

            count = 0
            for k in range(first_point + 1, last_point):
                count += 1
                profile[k] = i_low + count * delta

    elif edge_type == "roof":
        profile[:] = i_low
        if width == 1 or width == 3:
            profile[mid_idx] = i_high
        elif width == 2:
            profile[mid_idx : mid_idx + 2] = i_high
        else:
            delta = (i_high - i_low) / (width - np.ceil(width / 2))

            profile[first_point] = i_low
            profile[mid_idx] = i_high
            profile[last_point] = i_low

            # Left half
            count = 0
            for k in range(first_point + 1, mid_idx):
                count += 1
                profile[k] = i_low + count * delta

            # Right half
            count = 0
            for k in range(mid_idx + 1, last_point):
                count += 1
                profile[k] = i_high - count * delta

    else:
        raise ValueError(f"Unknown edge type: {edge_type}")

    # Generate 2D image
    # repmat(profile, M, 1) -> (M, N)
    # profile is (N,). Need to reshape to (1, N) then tile
    f = np.tile(profile.reshape(1, n), (m, 1))

    # Rotate
    if angle != 0:
        f = imageRotate4e(f, angle)

    # Crop
    # MATLAB:
    # lowX = M/4 + 1; highX = lowX + Morig - 1;
    # M is 2*Morig. M/4 is Morig/2.
    # lowX = Morig/2 + 1.
    # Python index: Morig//2.

    start_x = m // 4
    end_x = start_x + m_orig

    start_y = n // 4
    end_y = start_y + n_orig

    f = f[start_x:end_x, start_y:end_y]

    return f
