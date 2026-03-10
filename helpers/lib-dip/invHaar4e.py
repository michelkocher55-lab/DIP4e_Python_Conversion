import numpy as np

def invHaar4e(t):
    """
    Compute the inverse 1-D wavelet transform with respect to Haar wavelets.
    
    Parameters:
    -----------
    t : list or numpy.ndarray
        Haar transform coefficients. Length must be a power of 2.
        
    Returns:
    --------
    f : numpy.ndarray
        Reconstructed signal.
    """
    t = np.array(t, dtype=float)
    N_full = len(t)
    
    # Check power of 2
    log_n = np.log2(N_full)
    if not log_n.is_integer():
        print('The length of the transform must be a power of 2!')
        pass
        
    S = int(log_n)
    
    # In MATLAB, 't' is modified in place to hold intermediate reconstruction results.
    # We will do the same conceptual steps.
    
    # Working buffer (copy of input t)
    # The reconstruction starts at the coarsest level and works up.
    recon_buffer = t.copy()
    
    # Start loop
    # MATLAB: N=2 initially. s=1 to S.
    # c = sqrt(2)/2
    
    current_N = 2
    c = np.sqrt(2) / 2.0
    
    for s in range(S):
        # We process 'current_N' elements of recon_buffer.
        # But wait, input 't' structure:
        # [Approx | Detail_0 | Detail_1 | ... ]
        # The Approx is at t[0]. Detail_0 is t[1]. (For N=2 reconstruction).
        # We combine them to get 2 samples.
        
        # Determine half size (approximation part) and half size (detail part)
        # N=2. half=1.
        # indices 0..half-1 are approx.
        # indices half..N-1 are details.
        
        half_N = current_N // 2
        
        approx = recon_buffer[:half_N]
        details = recon_buffer[half_N:current_N]
        
        # Compute reconstruction for this level
        # f(i) = c * (approx + detail)
        # f(i+1) = c * (approx - detail)
        # We can interleave them.
        
        # We need a temporary array for these 'current_N' samples
        new_recon = np.zeros(current_N)
        
        # Vectorized scaling
        term1 = c * (approx + details)
        term2 = c * (approx - details)
        
        # Interleave
        # f[0::2] = term1
        # f[1::2] = term2
        new_recon[0::2] = term1
        new_recon[1::2] = term2
        
        # Update the buffer
        recon_buffer[:current_N] = new_recon
        
        # Prepare for next scale (grow N)
        current_N *= 2
        
    return recon_buffer
