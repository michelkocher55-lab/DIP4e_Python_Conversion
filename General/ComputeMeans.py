
import numpy as np
from General.qtgetblk import qtgetblk
from General.qtsetblk import qtsetblk

def ComputeMeans(I, S):
    """
    Compute mean of each block in quadtree decomposition.
    """
    means = I.astype(float) # Output image
    
    # Iterate dimensions used in decomposition (powers of 2)
    # TQTDecomp loop: [512 256 ... 1]
    # We should detect dimensions present in S or iterate standard range.
    # Dimensions present:
    if S.nnz == 0:
        return means
        
    dims = np.unique(S.data)
    
    for dim in dims:
        dim = int(dim) # Ensure int
        values = qtgetblk(I, S, dim)
        
        if values.size > 0:
            # values is (dim, dim, k)
            # Sum over dim, dim (axis 0, 1)
            # Result (k,)
            
            # doublesum = sum(sum(values,1,'double'),2); -> Sum over blocks
            # Here: sum axis 0 and 1
            block_sums = np.sum(values, axis=(0, 1))
            block_means = block_sums / (dim**2)
            
            # Repmat mean to fill block?
            # qtsetblk(means, S, dim, mean_val)
            # qtsetblk expects values of size (dim, dim, k).
            
            # Create (dim, dim, k) with constant mean
            k = values.shape[2]
            mean_blocks = np.zeros((dim, dim, k))
            for i in range(k):
                mean_blocks[:, :, i] = block_means[i]
                
            means = qtsetblk(means, S, dim, mean_blocks)
            
    return means
