import numpy as np

def haar4e(f):
    """
    Compute the N-point 1-D wavelet transform with respect to Haar wavelets
    using an averaging and differencing approach.
    
    Parameters:
    -----------
    f : list or numpy.ndarray
        1D Input signal. Length must be a power of 2.
        
    Returns:
    --------
    t : numpy.ndarray
        Transform coefficients.
    """
    f = np.array(f, dtype=float)
    N = len(f)
    
    # Check power of 2
    log_n = np.log2(N)
    if not log_n.is_integer():
        print('The length of the input function must be a power of 2!')
        # The MATLAB code prints a message but proceeds (likely to fail or do partial?)
        # For Python, let's stick to MATLAB behavior or raise error?
        # MATLAB loops S times. If S is not integer (e.g. 3.5), loop 1:3.5 is empty or floors?
        # Typically loop 1:3.5 runs 1, 2, 3.
        # But if N is not power of 2, the algorithm (N/2) logic fails for odd N.
        # Let's enforce it strictly or follow robust logic.
        # Given "f(i)+f(i+1)", it assumes pairs.
        # Let's assume strict power of 2 for correctness.
        pass # The loop below uses S from log2, which if float, int(S) matches floor.
    
    S = int(log_n)
    
    t = np.zeros(N)
    
    # We maintain 'f_curr' as the full vector being transformed
    f_curr = f.copy()
    
    current_N = N
    
    for s in range(S):
        # We need to process the first 'current_N' elements of f_curr
        # vector length for this scale
        half_N = current_N // 2
        
        # Extract signal part to transform
        sig_part = f_curr[:current_N]
        
        # Vectorized average and diff
        # sig_part[0::2] -> evens, sig_part[1::2] -> odds
        avgs = (sig_part[0::2] + sig_part[1::2]) / np.sqrt(2)
        diffs = (sig_part[0::2] - sig_part[1::2]) / np.sqrt(2)
        
        # Write back to f_curr (mimicking MATLAB's t)
        # First half gets averages
        f_curr[:half_N] = avgs
        # Second half gets differences
        f_curr[half_N:current_N] = diffs
        
        # The stored 'diffs' part is now final for this level.
        # The 'avgs' part becomes the input for the next level.
        # In MATLAB: f = t. 't' contains the updated values. 
        # Next loop iterates up to N/2.
        
        current_N = half_N
        
    return f_curr
