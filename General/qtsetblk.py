from typing import Any
from scipy import sparse


def qtsetblk(I: Any, S: Any, dim: Any, values: Any):
    """
    Set block values in a quadtree decomposition.

    Parameters:
    I (ndarray): Image to modify (or read size from).
    S (sparse matrix): Quadtree decomposition matrix (from qtdecomp).
    dim (int): Dimension of blocks to modify.
    values (ndarray): Values to put into the blocks.
                      Size should be (dim, dim, num_blocks).
                      If num_blocks is number of blocks of size dim.

    Returns:
    J (ndarray): Modified image.
    """
    # Create output
    J = I.copy()

    # Find blocks of size dim
    # S is sparse.
    if sparse.issparse(S):
        S = S.tocoo()

    # Indices where S == dim
    mask = S.data == dim
    rows = S.row[mask]
    cols = S.col[mask]

    num_blocks = len(rows)
    if num_blocks == 0:
        return J

    # Check values shape
    # values can be (dim, dim, num_blocks)
    # or (dim, dim) ?? (Repeated?)
    # "values is an array of size dim-by-dim-by-k, where k is the number of blocks."

    if values.ndim == 3 and values.shape[2] == num_blocks:
        # Loop blocks
        for i in range(num_blocks):
            r = rows[i]
            c = cols[i]
            block_val = values[:, :, i]

            # Constraints: block fits in J?
            # qtdecomp S corresponds to top-left.
            # J might be smaller than padded decomposition.
            # We clip.

            r_end = min(r + dim, J.shape[0])
            c_end = min(c + dim, J.shape[1])

            # If clipped, we slice values too
            h = r_end - r
            w = c_end - c

            J[r:r_end, c:c_end] = block_val[:h, :w]

    return J
