
import numpy as np
from scipy import ndimage as ndi

def bwboundaries(BW, conn=8):
    """
    Traces the boundaries of objects in binary image BW.
    Returns list of arrays (N x 2) [row, col], mimicking MATLAB bwboundaries.
    """
    BW = np.asarray(BW).astype(bool)
    if BW.ndim != 2:
        raise ValueError('BW must be a 2D binary image.')

    if conn == 4:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
        tracer = trace_boundary_4
    elif conn == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
        tracer = trace_boundary_8
    else:
        raise ValueError('conn must be 4 or 8.')

    labels, num = ndi.label(BW, structure=structure)
    if num == 0:
        return []

    boundaries = []
    # Label order follows row-major scan order, matching MATLAB-like output order.
    for k in range(1, num + 1):
        comp = (labels == k)
        bounded = np.pad(comp, 1, mode='constant', constant_values=0)

        rows, _ = np.where(bounded)
        if len(rows) == 0:
            continue
        min_r = np.min(rows)
        c_indices = np.where(bounded[min_r, :])[0]
        start_c = np.min(c_indices)
        start = (min_r, start_c)

        b = tracer(bounded, start)
        b = np.array(b, dtype=int) - 1
        boundaries.append(b)

    return boundaries

def trace_boundary_4(img, start):
    offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    boundary = [start]
    curr = start
    backtrack = 3 
    max_steps = img.size * 2
    steps = 0
    
    while True:
        steps += 1
        if steps > max_steps:
             break
        found = False
        for i in range(4):
            idx = (backtrack + 1 + i) % 4
            dr, dc = offsets[idx]
            nr, nc = curr[0] + dr, curr[1] + dc
            if img[nr, nc]:
                curr = (nr, nc)
                boundary.append(curr)
                found = True
                # Backtrack should be the opposite of the move direction.
                # Using idx+3 can bias the walk and skip valid outer turns
                # (e.g. small concave cases), producing short boundaries.
                backtrack = (idx + 2) % 4
                break
        if not found: break
        # Close when we return to the start after at least one move.
        # The previous condition (`backtrack == 0`) could leave open traces,
        # which then create apparent diagonal jumps when codes are computed.
        if curr == start and len(boundary) > 2:
            break

    # Remove duplicate consecutive points, if any.
    if len(boundary) > 1:
        compact = [boundary[0]]
        for p in boundary[1:]:
            if p != compact[-1]:
                compact.append(p)
        boundary = compact

    return boundary

def trace_boundary_8(img, start):
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    boundary = [start]
    curr = start
    backtrack = 6 
    max_steps = img.size * 2
    steps = 0
    
    while True:
        steps += 1
        if steps > max_steps:
             break
        found_next = False
        for i in range(8):
            idx = (backtrack + 1 + i) % 8
            dr, dc = offsets[idx]
            nr, nc = curr[0] + dr, curr[1] + dc
            if img[nr, nc]:
                curr = (nr, nc)
                boundary.append(curr)
                backtrack = (idx + 5) % 8
                found_next = True
                break
        if not found_next: break
        if curr == start and len(boundary) > 2: break
    return boundary
