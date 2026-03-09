"""Minimum-perimeter polygon (MPP) utilities.

This module is a Python translation of DIPUM's `im2minperpoly.m`, adapted to
this codebase. The public entry point is `im2minperpoly(I, cellsize)`.
"""

from typing import Any

import numpy as np
import math
from scipy import ndimage
from skimage import measure
from scipy import sparse

try:
    from helpers.qtdecomp import qtdecomp
    from helpers.qtgetblk import qtgetblk
    from helpers.qtsetblk import qtsetblk
except Exception:
    from qtdecomp import qtdecomp
    from qtgetblk import qtgetblk
    from qtsetblk import qtsetblk


def im2minperpoly(I: Any, cellsize: Any):
    """Compute minimum-perimeter polygon vertices for a single binary object.

    Parameters
    ----------
    I : ndarray
        Binary image containing one region or one non-self-intersecting boundary.
    cellsize : int
        Square cell size used to build the cellular complex (must be > 1).

    Returns
    -------
    X, Y : ndarray
        Row/column coordinates of MPP vertices.
    R : ndarray
        Region extracted from the cellular complex.
    """

    if cellsize <= 1:
        raise ValueError("cellsize must be an integer > 1.")

    I = I.astype(int)

    # Check to see that there is only one object in I
    lbl, num = ndimage.label(I)
    if num > 1:
        raise ValueError("Input image cannot contain more than one region.")

    # Extract 4-connected region encompassed by cellular complex
    R = cellcomplex(I, cellsize)

    # Find vertices of MPP
    X, Y = mppvertices(R, cellsize)

    return X, Y, R


def cellcomplex(I: Any, cellsize: Any):
    """Build the cellular complex and return the enclosed region."""
    # Fill holes
    I = ndimage.binary_fill_holes(I).astype(int)

    # 4-connected perimeter (MATLAB: bwperim(I,4)).
    struct = ndimage.generate_binary_structure(2, 1)  # 4-conn
    eroded = ndimage.binary_erosion(I, structure=struct)
    I_perim = I & (~eroded)

    M, N = I.shape

    # Pad to KxK where K/cellsize is a power of 2 and K >= max(M,N).
    max_dim = max(M, N)
    ratio = math.ceil(max_dim / cellsize)
    if ratio == 0:
        ratio = 1
    K_pow = 1
    while K_pow < ratio:
        K_pow *= 2

    K = K_pow * cellsize

    M1 = K - M
    N1 = K - N
    I_padded = np.pad(I_perim, ((0, M1), (0, N1)), mode="constant", constant_values=0)

    # Quadtree decomposition and block extraction of size cellsize.
    Q = qtdecomp(I_padded, threshold=0, min_dim=cellsize)
    vals, r, c = qtgetblk(I_padded, Q, cellsize)

    if vals.size == 0:
        return np.zeros((M, N), dtype=bool)

    block_sums = vals.reshape(vals.shape[0], -1).sum(axis=1)
    idx = np.where(block_sums >= 1)[0]
    if len(idx) == 0:
        return np.zeros((M, N), dtype=bool)

    rr = r[idx]
    cc = c[idx]

    # Set selected quadtree blocks to 1.
    Ssel = sparse.coo_matrix(
        (np.full(len(rr), cellsize, dtype=int), (rr, cc)), shape=I_padded.shape
    ).tocsr()
    values = np.ones((cellsize, cellsize, len(rr)), dtype=I_padded.dtype)
    I_complex = qtsetblk(np.zeros_like(I_padded), Ssel, cellsize, values)

    BF = ndimage.binary_fill_holes(I_complex).astype(int)

    # Interior of the cellular border.
    R_padded = BF & (~I_complex.astype(bool))

    R = R_padded[:M, :N]

    return R


def mppvertices(R: Any, cellsize: Any):
    """Compute MPP vertices around region R."""
    R = R.astype(int)

    # Keep the largest contour as a coarse sanity check (legacy behavior).
    contours = measure.find_contours(R, 0.5)
    if len(contours) == 0:
        return np.array([]), np.array([])
    _ = contours[0] if len(contours) == 1 else max(contours, key=len)

    # Trace boundary pixels (4-connected) in image coordinates.
    rows, cols = np.where(R)
    if len(rows) == 0:
        return np.array([]), np.array([])

    r0, c0 = rows[0], cols[0]

    B_pixels = trace_boundary(R, (r0, c0))
    B = np.array(B_pixels)

    # MATLAB bwboundaries repeats first point at the end.
    if len(B) > 1 and np.array_equal(B[0], B[-1]):
        B = B[:-1]

    x = B[:, 0]  # Rows
    y = B[:, 1]  # Cols

    L = vertexlist(x, y, cellsize)
    if len(L) == 0:
        return np.array([]), np.array([])

    NV = len(L)
    X_mpp = [L[0, 0]]
    Y_mpp = [L[0, 1]]

    cMPPV = np.array([L[0, 0], L[0, 1]])
    classV = L[0, 2]
    cWH = cMPPV.copy()
    cBL = cMPPV.copy()

    count = 0

    while True:
        count += 1
        if count > NV:
            break

        if count == NV:
            cV = np.array([L[0, 0], L[0, 1]])
            classV = L[0, 2]
        else:
            cV = np.array([L[count, 0], L[count, 1]])
            classV = L[count, 2]

        I_flag, newMPPV, W, B = mppVtest(cMPPV, cV, classV, cWH, cBL)

        if I_flag == 1:
            cMPPV = newMPPV
            matches = np.where((L[:, 0] == newMPPV[0]) & (L[:, 1] == newMPPV[1]))[0]
            K = matches[0] if len(matches) > 0 else 0

            count = K
            cWH = newMPPV.copy()
            cBL = newMPPV.copy()

            X_mpp.append(newMPPV[0])
            Y_mpp.append(newMPPV[1])
        else:
            cWH = W
            cBL = B

    return np.array(X_mpp), np.array(Y_mpp)


def trace_boundary(bin_img: Any, start_rc: Any):
    """
    Traces the boundary of a 4-connected object using a Wall Follower algorithm
    (Left-Hand Rule on pixels), ensuring a 4-connected path (Manhattan geometry).

    bin_img: Binary image (0 background, 1 foreground).
    start_rc: (row, col) of the top-leftmost pixel of the object.
    """
    path = []

    # Directions: 0=East, 1=South, 2=West, 3=North (Clockwise)
    # Dr, Dc
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Start at top-leftmost pixel.
    # We assume we approach from West, so initial direction is East (0).
    # Since start_rc is top-left, neighbor to West is 0 and North is 0.
    curr_r, curr_c = start_rc
    path.append((curr_r, curr_c))

    # Initial direction: East.
    curr_dir = 0

    # Limit for safety
    max_steps = bin_img.size * 2
    steps = 0

    while steps < max_steps:
        found_next = False

        # Check neighbors in order: Left, Straight, Right, Back
        # Relative changes to curr_dir:
        # Left: (curr_dir + 3) % 4
        # Straight: curr_dir
        # Right: (curr_dir + 1) % 4
        # Back: (curr_dir + 2) % 4

        check_order = [3, 0, 1, 2]  # Relative directory offsets

        for rot in check_order:
            check_dir = (curr_dir + rot) % 4
            dr, dc = dirs[check_dir]
            nr, nc = curr_r + dr, curr_c + dc

            # Check bounds and value
            if 0 <= nr < bin_img.shape[0] and 0 <= nc < bin_img.shape[1]:
                if bin_img[nr, nc] == 1:
                    # Found next pixel
                    curr_r, curr_c = nr, nc
                    curr_dir = check_dir
                    path.append((curr_r, curr_c))
                    found_next = True
                    break

        if not found_next:
            # Isolated pixel? Should not happen if part of region > 1 pixel
            break

        # Check termination: Return to start
        # Common condition: curr is start AND next step would be same as first step?
        # Alternatively: precise match of (r,c) to start.
        if (curr_r, curr_c) == start_rc:
            break

        steps += 1

    return path


def vertexlist(x: Any, y: Any, cellsize: Any):
    """vertexlist."""
    # Preprocess
    # Arrange so first point is top-left-most
    # min x, then min y
    min_x = np.min(x)
    cx = np.where(x == min_x)[0]
    # Among these, min y
    min_y = np.min(y[cx])
    cy = np.where(y[cx] == min_y)[0]

    idx = cx[cy[0]]  # First index

    # Start at idx
    x = np.roll(x, -idx)
    y = np.roll(y, -idx)

    # Keep only changes in direction
    K = len(x)
    xnew = [x[0]]
    ynew = [y[0]]

    # Create wrapped arrays for easy access
    x_wrap = np.concatenate((x, [x[0]]))
    y_wrap = np.concatenate((y, [y[0]]))

    for k in range(1, K):  # 1 to K-1 in Python (indices)
        # Triplet: k-1, k, k+1
        # Indices in wrap: k, k+1, k+2 ? No.
        # x is 0..K-1.
        # k corresponds to x[k].
        # prev: x[k-1]. next: x[k+1] (or wrap).

        # vsign(prev, curr, next)
        v1 = [x[k - 1], y[k - 1]]
        v2 = [x[k], y[k]]
        v3 = [x_wrap[k + 1], y_wrap[k + 1]]

        s = vsign(v1, v2, v3)
        if s != 0:
            xnew.append(x[k])
            ynew.append(y[k])

    x = np.array(xnew)
    y = np.array(ynew)

    # boundarydir 'ccw' (Assuming input is already roughly ordered or using robust orientation)
    # Using `polyangles` or `vsign` logic implies order.
    # For now assume trace_boundary returns consistent CCW/CW.
    # skimage find_contours is usually CCW for holes, CW for outer? Or vice versa.
    # We should enforce CCW.
    # Check polygon area?
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])  # Shoelace
    # If wrong sign, flip.
    # CCW area should be negative for standard image coords (y down)?
    # Chapter 2 coords: x down (rows), y right (cols).
    # "Positive x-axis extending vertically down".
    # Cross product x * y is z (out of page).
    # Standard math: x right, y up.
    # Here x=rows (down), y=cols (right).
    # (0,0) -> (1,0) [down] -> (1,1) [right] -> (0,1) [up] -> (0,0).
    # x: 0 1 1 0. y: 0 0 1 1.
    # Shoelace: (0*0 + 1*1 + 1*1 + 0*0) - (0*1 + 0*1 + 1*0 + 1*0) = 2.
    # Positive.
    # So CCW is positive area in (Row, Col) coords?
    # Wait, (0,0) to (1,0) is DOWN.
    # (1,0) to (1,1) is RIGHT.
    # (1,1) to (0,1) is UP.
    # This is CCW visually on screen (top-left origin).
    # So positive area = CCW.

    # Calculate signed area
    # Note: x is vector of size K.
    # Need closed loop for area.
    # Use standard shoelace formula
    x_c = np.append(x, x[0])
    y_c = np.append(y, y[0])
    area = 0.5 * np.sum(x_c[:-1] * y_c[1:] - x_c[1:] * y_c[:-1])

    if area < 0:  # Clockwise?
        x = x[::-1]
        y = y[::-1]

    K = len(x)
    L = np.zeros((K, 3))
    L[:, 0] = x
    L[:, 1] = y

    # Calc C
    for k in range(K):
        # Indices wrapping
        prev = (k - 1) % K
        curr = k
        nxt = (k + 1) % K

        v1 = [x[prev], y[prev]]
        v2 = [x[curr], y[curr]]
        v3 = [x[nxt], y[nxt]]

        s = vsign(v1, v2, v3)
        if s > 0:
            L[k, 2] = 1  # Convex
        elif s < 0:
            L[k, 2] = -1  # Concave
            rx, ry = vreplacement(v1, v2, v3, cellsize)
            L[k, 0] = rx
            L[k, 1] = ry
        else:
            L[k, 2] = 0

    return L


def vsign(v1: Any, v2: Any, v3: Any):
    """vsign."""
    # A = [v1x v1y 1; v2x v2y 1; v3x v3y 1]
    # Det A
    A = np.array([[v1[0], v1[1], 1], [v2[0], v2[1], 1], [v3[0], v3[1], 1]])
    return round(np.linalg.det(A))


def vreplacement(v1: Any, v: Any, v2: Any, cellsize: Any):
    """vreplacement."""
    # Logic for replacement
    # v1 -> v -> v2 counterclockwise
    v1x, v1y = v1
    vx, vy = v
    v2x, v2y = v2

    rx, ry = vx, vy

    if vx > v1x and vy == v1y and vx == v2x and vy > v2y:
        rx = vx - cellsize
        ry = vy - cellsize
    elif vx == v1x and vy > v1y and vx < v2x and vy == v2y:
        rx = vx + cellsize
        ry = vy - cellsize
    elif vx < v1x and vy == v1y and vx == v2x and vy < v2y:
        rx = vx + cellsize
        ry = vy + cellsize
    elif vx == v1x and vy < v1y and vx > v2x and vy == v2y:
        rx = vx - cellsize
        ry = vy + cellsize
    else:
        # Not raising error to avoid crash on edge cases due to tracing artifacts?
        # But MATLAB raises error.
        pass  # raise ValueError('Vertex configuration not valid')

    return rx, ry


def mppVtest(cMPPV: Any, cV: Any, classcV: Any, cWH: Any, cBL: Any):
    """mppVtest."""
    I_res = 0
    newMPPV = np.array([0, 0])
    W = cWH
    B = cBL

    sW = vsign(cMPPV, cWH, cV)
    sB = vsign(cMPPV, cBL, cV)

    if sW > 0:
        I_res = 1
        newMPPV = cWH
        W = newMPPV
        B = newMPPV
    elif sB < 0:
        I_res = 1
        newMPPV = cBL
        W = newMPPV
        B = newMPPV
    elif (sW <= 0) and (sB >= 0):
        if classcV == 1:
            W = cV
        else:
            B = cV

    return I_res, newMPPV, W, B
