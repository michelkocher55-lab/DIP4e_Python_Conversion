from typing import Any
import numpy as np
from scipy import sparse


def qtdecomp(I: Any, threshold: Any = None, min_dim: Any = 1, predicate: Any = None):
    """
    Quadtree decomposition.

    Parameters:
    I : ndarray
        Input image (grayscale). Assumed square and dimensions power of 2 (or handled by caller).
    threshold : float or None
        Threshold for splitting (if no predicate provided).
        Standard MATLAB qtdecomp(I, threshold) splits if max(block) - min(block) > threshold.
    min_dim : int
        Minimum dimension of a block.
    predicate : callable
        Function that accepts a block (or list of blocks) and returns bools.
        If provided, threshold is ignored (or passed to predicate if it expects it).
        Signature: predicate(block) -> bool

    Returns:
    S : scipy.sparse.lil_matrix
        Sparse matrix where S[r, c] = dim indicates a block of size dim x dim starting at (r, c).
    """

    I = np.asarray(I)
    M, N = I.shape

    # Initialize Sparse Matrix
    # We use LIL for efficient setting of individual elements
    S = sparse.lil_matrix((M, N), dtype=int)

    # Default predicate if none provided: Range > threshold
    if predicate is None:
        if threshold is None:
            threshold = 0

        def predicate(block: Any):
            """predicate."""
            return (block.max() - block.min()) > threshold

    def split(r: Any, c: Any, dim: Any):
        """split."""
        block = I[r : r + dim, c : c + dim]

        # If block is too small, stop splitting, record size
        if dim <= min_dim:
            S[r, c] = dim
            return

        # Check predicate
        # Optimization: We can check predicate on the whole block.
        # If predicate says SPLIT:
        should_split = predicate(block)

        if should_split:
            # Split into 4
            half = dim // 2
            split(r, c, half)
            split(r, c + half, half)
            split(r + half, c, half)
            split(r + half, c + half, half)
        else:
            # No split, record this block
            S[r, c] = dim

    # Start recursion
    # Ensure square power of 2? MATLAB qtdecomp does not strictly enforce padding inside,
    # but splitmerge pads carefully. We assume caller handles padding if strict power of 2 is needed.
    # But basic qtdecomp works on any rectangular? No, usually square power of 2 is standard for easy quadtree.
    # splitmerge.m handles padding. We will assume full image start.

    # Handling non-square or non-power-of-2 input in recursion might be tricky with fixed halving.
    # We'll assume the standard usage where it's already padded or we just process the max square?
    # For now, start with full image.
    split(0, 0, M)

    return S.tocsr()  # Return CSR for general usage or stay sparse
