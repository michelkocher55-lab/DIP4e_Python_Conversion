from typing import Any
import numpy as np


def boundaryTracer4e(f: Any, direction: Any = "cw"):
    """
    Traces the boundaries of objects in an image.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image. Can be binary or labeled image.
        If binary (0s and 1s), it is treated as one or unconnected objects of label 1.
        If labeled (integers), each integer > 0 is treated as a separate object.
    direction : str, optional
        'cw' (clockwise, default) or 'ccw' (counter-clockwise).

    Returns:
    --------
    B : list of numpy.ndarray
        List of boundaries. Each element is an (NP, 2) array of (row, col) coordinates.
    """
    f = np.array(f, dtype=int)

    # Pad input to handle boundaries touching the image edge
    # imPad4e(f, pad_size, val=0) -> assumes 'zeros' padding by default (val=0 implicitly for 'both')
    # MATLAB: L = imPad4e(L, 1, 1); -> pad 1 pixel, value implied?
    # Let's use mode='constant' (zeros) implicitly if not specified?
    # imPad4e signature: f, padsize, method, direction
    # MATLAB call: imPad4e(L, 1, 1)? Wait, MATLAB imPad4e args are likely different.
    # The snippet says: "Lp = imPad4e(L,1,1);"
    # My python imPad4e: def imPad4e(f, pad_width, method='constant', ...)?
    # Let's check imPad4e.py if possible.
    # If I just use np.pad it's safer for this logic, but I should use the utility if "project function".
    # I'll stick to np.pad for internal reliability here allowing independent verification?
    # No, I should use imPad4e for "Code Re-use".
    # Let's assume imPad4e accepts (f, [1,1]) or similar.
    # Actually, simplistic:

    Lp = np.pad(f, 1, mode="constant", constant_values=0)

    # Preallocate output list
    # Number of objects = max label
    num_objects = np.max(Lp)
    if num_objects == 0:
        return []

    B = [None] * num_objects  # To hold outcome for each label index

    # Loop over objects
    # Note: Labeled images are usually 1..N.
    # Binary image max is 1.

    for k in range(1, num_objects + 1):
        # Extract object mask
        current_object = (Lp == k).astype(int)

        # Trace boundary
        # If object doesn't exist (gap in labels), skip?
        if np.sum(current_object) == 0:
            B[k - 1] = np.empty((0, 2), dtype=int)
            continue

        bout = _traceBoundary(current_object, direction)

        # Undo padding (subtract 1 from all coords)
        bout = bout - 1

        B[k - 1] = bout

    return B


def _traceBoundary(I: Any, direction: Any = "cw"):
    """
    Traces the boundary of a single object in binary image I.
    """
    if direction not in ["cw", "ccw"]:
        direction = "cw"

    # Find starting point (uppermost-leftmost)
    # np.argwhere returns (row, col) sorted by row then col.
    # So the first element is exactly what we want.
    points = np.argwhere(I == 1)
    if len(points) == 0:
        return np.empty((0, 2), dtype=int)

    # Uppermost (min row), then Leftmost (min col)
    # argwhere is sorted by axis 0 then 1, so `points[0]` is min_row, min_col_for_that_row
    # which is exactly uppermost-leftmost.
    start_point = points[0]
    xb0, yb0 = start_point[0], start_point[1]

    # Initialize trace
    b0 = np.array([xb0, yb0])
    xbold, ybold = xb0, yb0

    # List to collect boundary points
    bout = [b0]

    current_point = False

    # Neighbors of (xb0, yb0)
    # Order: [W, NW, N, NE, E, SE, S, SW] (indices 0 to 7)
    b_nhood = _coord2nhood(I, xb0, yb0)

    # c0 is W neighbor of b0. W is index 0.
    c_nhood = np.zeros(8, dtype=int)
    c_nhood[0] = 1  # W is 1

    while not current_point:
        # Find index of 1 in c_nhood
        # Should contain exactly one 1.
        idc_list = np.where(c_nhood == 1)[0]
        if len(idc_list) == 0:
            break  # Should not happen
        idc = idc_list[0]

        # Find indices of 1s in b_nhood
        idb_list = np.where(b_nhood == 1)[0]

        if len(idb_list) == 0:
            # Single pixel object?
            current_point = True
            break

        # Find 'id': first 1 in b_nhood scanning CW from idc
        # Logic: look for index > idc. If none, wrap around (index < idc).

        # Indices in idb_list greater than idc
        greater = idb_list[idb_list > idc]
        if len(greater) > 0:
            id_next = greater[0]
        else:
            # Wrap around: smallest index
            id_next = idb_list[0]

        # Update neighborhoods
        # Find coordinates of b_new (at id_next)
        xbnew, ybnew = _nhood2coord(xbold, ybold, id_next)

        # Update b_nhood for next step
        b_nhood = _coord2nhood(I, xbnew, ybnew)

        # Update c_nhood
        # c_new is the point just before b_new in scan order around b_old.
        # id_next is b_new's index. c_new's index relative to b_old is id_next - 1 (with wrap)
        # Note: MATLAB code `id-1`. If `id=1` (index 1 is W in MATLAB, index 0 in Python),
        # MATLAB logic: `if id==1, id-1=8`.
        # Python indices 0..7.
        # If id_next=0 (W), precedent is 7 (SW).
        # So `(id_next - 1) % 8`.

        id_c_rel_bold = (id_next - 1) % 8
        xcnew, ycnew = _nhood2coord(xbold, ybold, id_c_rel_bold)

        # `point2nhood` calculates where (xcnew, ycnew) is relative to (xbnew, ybnew)
        c_nhood = _point2nhood(xcnew, ycnew, xbnew, ybnew)

        # Check if done
        if xbnew == xb0 and ybnew == yb0:
            current_point = True
        else:
            bout.append(np.array([xbnew, ybnew]))

        # Update for next pass
        xbold, ybold = xbnew, ybnew

        # Safety break to prevent infinite loops in bad geometry
        if len(bout) > I.size:
            break

    bout = np.array(bout)

    # Handle CCW
    # "reverse the order ... leaving starting point the same"
    if direction == "ccw" and len(bout) > 1:
        # Keep bout[0], reverse bout[1:]
        bout[1:] = bout[1:][::-1]

    return bout


# Lookup tables for [W, NW, N, NE, E, SE, S, SW] => indices 0..7
# (row_offset, col_offset)
# W: (0, -1)
# NW: (-1, -1)
# N: (-1, 0)
# NE: (-1, 1)
# E: (0, 1)
# SE: (1, 1)
# S: (1, 0)
# SW: (1, -1)

X_OFFSETS = [0, -1, -1, -1, 0, 1, 1, 1]
Y_OFFSETS = [-1, -1, 0, 1, 1, 1, 0, -1]


def _coord2nhood(I: Any, xc: Any, yc: Any):
    """
    Values of 8 neighbors: [W, NW, N, NE, E, SE, S, SW]
    """
    nhood = np.zeros(8, dtype=int)
    for i in range(8):
        nx = xc + X_OFFSETS[i]
        ny = yc + Y_OFFSETS[i]
        # Check bounds? I is padded, but check safety
        if 0 <= nx < I.shape[0] and 0 <= ny < I.shape[1]:
            nhood[i] = I[nx, ny]
    return nhood


def _nhood2coord(xold: Any, yold: Any, idx: Any):
    """
    Coordinates of neighbor at index idx.
    """
    xnew = xold + X_OFFSETS[idx]
    ynew = yold + Y_OFFSETS[idx]
    return xnew, ynew


def _point2nhood(xp: Any, yp: Any, xc: Any, yc: Any):
    """
    Returns one-hot array of location of (xp, yp) relative to (xc, yc).
    """
    dx = xp - xc
    dy = yp - yc

    idx = -1
    for i in range(8):
        if X_OFFSETS[i] == dx and Y_OFFSETS[i] == dy:
            idx = i
            break

    nhood = np.zeros(8, dtype=int)
    if idx != -1:
        nhood[idx] = 1
    return nhood
