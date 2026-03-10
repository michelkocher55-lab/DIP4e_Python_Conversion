import numpy as np

def morphoMatchLoops4e(I, B, padval=0, mode='same'):
    """
    Finds matches of SE B in binary image I using loops (Demonstration).
    
    S = morphoMatchLoops4e(I, B, padval=0, mode='same')
    
    Parameters
    ----------
    I : numpy.ndarray
        Binary image.
    B : numpy.ndarray
        Structuring element. Values: 0, 1, or other (don't care).
    padval : int
        Padding value (0 or 1).
    mode : str
        'same' or 'full'.
        
    Returns
    -------
    S : numpy.ndarray
        Match array (0: None, 0.5: Partial, 1.0: Perfect).
    """
    
    # Pre-processing
    I = np.array(I).astype(float)
    I[I > 0] = 1 # Ensure binary
    
    B = np.array(B)
    
    M, N = I.shape
    m, n = B.shape
    
    # Padding
    Ip = np.pad(I, ((m, m), (n, n)), mode='constant', constant_values=padval)
    
    # Don't care indices
    # B vals not 0 and not 1
    idx = np.where((B != 0) & (B != 1))
    
    S = np.zeros_like(Ip)
    
    # Centers of B
    Bxc = m // 2
    Byc = n // 2
    
    limit_x = M + m - 1
    limit_y = N + n - 1
    
    for x in range(limit_x):
        for y in range(limit_y):
            # Ipsub
            Ipsub = Ip[x : x+m, y : y+n]
            
            if Ipsub.shape != B.shape:
                continue
                
            xc = x + Bxc
            yc = y + Byc
            
            Btemp = B.copy()
            
            # Handle Don't Care
            if len(idx[0]) > 0:
                Btemp[idx] = Ipsub[idx]
                
            # Compare
            allMatches = (Btemp == Ipsub)
            C = np.sum(allMatches)
            
            if C == m * n:
                S[xc, yc] = 1.0
            elif 0 < C < m * n:
                S[xc, yc] = 0.5
                
    # Crop
    if mode == 'same':
        S = S[m : M+m, n : N+n]
        
    return S
