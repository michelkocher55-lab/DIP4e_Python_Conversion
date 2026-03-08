
import numpy as np

def qtgetblk(I, S, dim):
    """
    Get block values from quadtree decomposition.
    
    Parameters:
    I : ndarray
        The image.
    S : sparse matrix
        The quadtree structure (from qtdecomp). S[r, c] = block_size.
    dim : int
        The block dimension to retrieve.
        
    Returns:
    vals : ndarray
        Array of blocks of size (K, dim, dim).
    r : ndarray
        Row coordinates of top-left corners.
    c : ndarray
        Col coordinates of top-left corners.
    """
    
    # Find coordinates where S == dim
    # S is sparse.
    # Convert to COO to easily find (row, col, data)
    S_coo = S.tocoo()
    
    mask = (S_coo.data == dim)
    r = S_coo.row[mask]
    c = S_coo.col[mask]
    
    if len(r) == 0:
        return np.array([]), np.array([]), np.array([])
        
    K = len(r)
    vals = np.zeros((K, dim, dim), dtype=I.dtype)
    
    for i in range(K):
        rr, cc = r[i], c[i]
        vals[i] = I[rr:rr+dim, cc:cc+dim]
        
    return vals, r, c
