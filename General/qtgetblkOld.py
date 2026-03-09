from typing import Any
import numpy as np
from scipy import sparse


def qtgetblk(I: Any, S: Any, dim: Any):
    """
    Get block values from a quadtree decomposition.

    Parameters:
    I (ndarray): Image.
    S (sparse matrix): Quadtree decomposition.
    dim (int): Dimension of blocks to retrieve.

    Returns:
    vals (ndarray): Array of size (dim, dim, k) where k is the number of blocks.
                    Returns empty array if no blocks found.
    """
    if sparse.issparse(S):
        S = S.tocoo()

    mask = S.data == dim
    rows = S.row[mask]
    cols = S.col[mask]

    num_blocks = len(rows)
    if num_blocks == 0:
        return np.array([])

    vals = np.zeros((dim, dim, num_blocks), dtype=I.dtype)

    for i in range(num_blocks):
        r = rows[i]
        c = cols[i]

        # Extract block (handling boundary clipping if I is smaller than S implied size)
        # Assuming S matches I
        r_end = min(r + dim, I.shape[0])
        c_end = min(c + dim, I.shape[1])

        h = r_end - r
        w = c_end - c

        # If clipped, we might need padding logic or return partial?
        # MATLAB qtgetblk usually returns full blocks.
        # "If I is padded, blocks are full size".
        # We fill what we have.
        vals[:h, :w, i] = I[r:r_end, c:c_end]

    return vals
