import numpy as np

def entropy4e(f, n=256):
    """
    Computes a first-order estimate of the entropy of a matrix.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input matrix/image.
    n : int, optional
        Number of bins for histogram. Default 256.
        
    Returns:
    --------
    e : float
        Entropy value in bits.
    """
    f = np.array(f)
    
    # Flatten array
    f_flat = f.ravel()
    
    # Compute histogram
    # MATLAB: hist(double(f(:)), n)
    # Note: MATLAB's 'hist' uses n bins centered appropriately or assumes range.
    # For images, if we specify n=256, it usually implies covering the range.
    # numpy.histogram returns edges.
    # If we want simple bin counts, we can use density=True to get probability density directly 
    # but we need discrete probabilities.
    
    # Simply using histogram with n bins over the range of data? 
    # Or strict integers? The MATLAB function handles 'double'.
    
    # We'll use the full range of the data if not specified, 
    # or rely on standard image ranges.
    # Actually, for general matrices, np.histogram(f, bins=n) is robust.
    
    hist_counts, _ = np.histogram(f_flat, bins=n)
    
    # Compute probabilities
    # Normalize by total count
    p = hist_counts / np.sum(hist_counts)
    
    # Remove zeros (log2(0) is -inf)
    p = p[p > 0]
    
    # Compute entropy
    # e = -sum(p * log2(p))
    e = -np.sum(p * np.log2(p))
    
    return e
