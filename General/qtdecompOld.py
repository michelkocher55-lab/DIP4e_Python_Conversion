from typing import Any
import numpy as np
from scipy import sparse


def qtdecomp(I: Any, threshold: Any, min_dim: Any = 1, max_dim: Any = None):
    """
    Quadtree decomposition.

    Parameters:
    I (ndarray): Input image (grayscale).
    threshold (float): Threshold for splitting.
                       If max(block) - min(block) <= threshold, the block is uniform.
    min_dim (int): Minimum block dimension (must be power of 2).
    max_dim (int): Maximum block dimension (must be power of 2).
                   If None, defaults to the largest power of 2 <= min(I.shape).

    Returns:
    S (sparse matrix): Sparse matrix where S[i,j] = dim if a block of size dim starts at (i,j).
    """
    I = np.array(I)
    rows, cols = I.shape

    # Ensure square, power of 2? MATLAB qtdecomp works on any size but pads?
    # Usually works on square or pads to square.
    # Let's handle square requirement for simplicity or pad.
    # MATLAB qtdecomp: "If I is m-by-n, qtdecomp pads it with zeros to size k-by-k so that k is the next power of two."
    # We should reproduce this validation or padding.

    # Simple recursive approach

    # Determine size
    M, N = rows, cols
    sz = 2 ** np.ceil(np.log2(max(M, N))).astype(int)

    # Pad if necessary
    I_padded = np.zeros((sz, sz), dtype=I.dtype)
    I_padded[:M, :N] = I

    # Sparse matrix for result (using keys for now)
    blocks = {}  # (row, col) -> size

    if max_dim is None:
        max_dim = sz

    def decompose(r: Any, c: Any, dim: Any):
        """decompose."""
        # r, c are top-left coordinates in I_padded

        # Check decomposition criteria
        # If dim <= min_dim, stop
        if dim <= min_dim:
            blocks[(r, c)] = dim
            return

        # Get block
        block = I_padded[r : r + dim, c : c + dim]

        # Check homogeneity
        # Note: If block is partly outside original image, we should probably handle it.
        # But padding with 0s handles it if we consider 0s as part of image.
        # But usually we only care about original region.
        # MATLAB includes padded regions in decomposition.

        # Criterion: max - min <= threshold
        # If I is float, threshold is float. If I is uint8, threshold 0..1 or 0..255?
        # MATLAB: If I is uint8, threshold is in 0..1 range? Or 0..255?
        # "If I is of class uint8, threshold is multiplied by 255."
        # We'll assume threshold matches image scale or user handles it.

        vals = block
        if np.max(vals) - np.min(vals) <= threshold:
            blocks[(r, c)] = dim
        else:
            # Split
            new_dim = dim // 2
            decompose(r, c, new_dim)
            decompose(r, c + new_dim, new_dim)
            decompose(r + new_dim, c, new_dim)
            decompose(r + new_dim, c + new_dim, new_dim)

    decompose(0, 0, max_dim)

    # Convert blocks dict to sparse matrix (S)
    # S has size of I (or I_padded?)
    # MATLAB returns S of size I.
    # If blocks are outside I, they are discarded or included?
    # MATLAB returns S same size as I.
    # So we filter blocks starting inside I?
    # Or S is size(I_padded) but cropped?
    # "S is a sparse matrix of size size(I)."

    S_data = []
    S_rows = []
    S_cols = []

    for (r, c), dim in blocks.items():
        if r < M and c < N:
            S_rows.append(r)
            S_cols.append(c)
            # If block extends beyond M, N, is it reported?
            # Yes, acts as if image was padded, but S is cropped.
            # But the block SIZE is reported.
            S_data.append(dim)

    # Create CSR or COO
    # shape (M, N)
    S = sparse.coo_matrix((S_data, (S_rows, S_cols)), shape=(M, N))
    return S
