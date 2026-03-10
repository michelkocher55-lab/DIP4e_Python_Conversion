import numpy as np
import matplotlib.pyplot as plt

def wave2gray4e(c, s, scale=1, border='absorb'):
    """
    Displays wavelet decomposition coefficients.
    
    w = wave2gray4e(c, s, scale=1, border='absorb')
    
    Parameters
    ----------
    c : numpy.ndarray
        Wavelet decomposition vector (1D).
    s : numpy.ndarray
        Bookkeeping matrix (S).
        S(0) : Size of approx coeffs at level N
        S(1)..S(N) : Size of details at level N..1
        S(N+1) : Size of original image
    scale : int
        Detail scaling.
        0 or 1: Default (0 is auto max range? Code says 0->1).
        >1: Magnify default by scale factor.
        <0: Magnify absolute values by abs(scale).
    border : str
        'absorb' (default), 'append', 'none'.
        
    Returns
    -------
    w : numpy.ndarray
        The constructed image.
    """
    
    c = np.array(c).flatten()
    s = np.array(s)
    
    # Defaults
    if scale == 0:
        scale = 1
    
    absflag = scale < 0
    scale = abs(scale)
    
    # 1. Extract Approximation (Level N, first decomposition)
    # The first row of S is size of Approximation.
    # Level N (Coarsest) is at the beginning of C?
    # MATLAB wavedec2: [AN, HN, VN, DN, ..., H1, V1, D1].
    
    # Approximation AN is at start.
    # Size from S[0].
    # But wait, Helper function wavecut implementation needed.
    # Let's define helper to slice C based on S.
    
    cd_all = c # Use full vector for extraction helpers
    
    # Extract Approximation
    # 'a' (Approximation) is always at the beginning.
    # But `wavecut('a', c, s)` gets the Approximation?
    # Actually, often `wavecut` gets Detail.
    # `wavecut('a', c, s)` gets AN.
    
    # Implement slices manually.
    # Structure of C:
    # Segment 0: AN. Size s[0].
    # Segment 1: HN. Size s[1] (Usually same as s[0]).
    # Segment 2: VN. 
    # Segment 3: DN.
    # Segment 4: H(N-1). Size s[2].
    # ...
    
    # S has N+2 rows.
    # Level N (Coarsest) corresponds to index 1 (in MATLAB loop i=size(S,1)-2 ... 1).
    # MATLAB Loop goes from i = (N) down to 1.
    # Wait, size(s,1) = N+2.
    # Loop `i = size(S,1) - 2` corresponds to N.
    # Example: 2 Levels. S has 4 rows.
    # Rows: [szA2], [szD2], [szD1], [szImg].
    # i goes 2 down to 1.
    
    # Extract AN (w in code)
    # "wavecut('a', c, s)"
    # This corresponds to the FIRST block.
    # Size s[0].
    n_approx = s[0,0] * s[0,1]
    an = c[:n_approx].reshape(s[0])
    
    # Initialize w with normalized AN
    w = mat2gray(an)
    
    # Calculate global max detail for scaling?
    # Code: "cdx = max(abs(cd(:))) / scale".
    # Here `cd` comes from `wavecut('a'...)`?
    # Wait. MATLAB Line 54: `[cd, w] = wavecut('a', c, s); w = mat2gray(w);`
    # What does `wavecut` return? `[cd, w]`?
    # Usually `wavecut` zeroes out coefficients in C? Or returns the coefficients?
    # If `wavecut` modifies coefficients, then `cd` might be the REST of C?
    # Re-reading line 54 carefully:
    # `[cd, w] = wavecut('a', c, s);`
    # If `cd` is `[H, V, D...]`? and `w` is `A`?
    # Actually, `cdx` calculation uses `cd`.
    # `cdx = max(abs(cd(:))) / scale`.
    # This likely uses ALL DETAIL coefficients to find max range.
    # So `cd` implies "Coefficients Detail"?
    
    # Let's assume `wavecut('a', ...)` separates Approximation (w) from Details (cd).
    # So `cd` contains all H, V, D parts. `w` contains A.
    
    # So I need to split C into [AN] and [Rest].
    details_vec = c[n_approx:]
    
    # Calculate Scale Factor
    cdx = np.max(np.abs(details_vec)) / scale
    if cdx == 0: cdx = 1 # Prevent div/0
    
    # Scaling Details
    # If absflag, map [0, cdx] -> [0, 1].
    # Else map [-cdx, cdx] -> [0, 1].
    
    # We will process details level by level in the loop.
    # We should normalize them as we extract them using `cdx`.
    
    fill = 0 if absflag else 0.5
    
    # Loop i from N down to 1
    # MATLAB (1-based): size(s,1)-2 down to 1.
    # Number of levels N_levels = size(s,1) - 2.
    N_levels = s.shape[0] - 2
    
    current_idx_c = n_approx # Pointer to consume details
    
    # We need to extract levels in order: N, N-1, ... 1.
    # C layout: AN, (HN, VN, DN), (HN-1, VN-1, DN-1), ...
    # So we simply iterate forward in C.
    
    # But MATLAB Loop usually goes from Largest Scale (Coarsest, N) down to Finest (1).
    # And we build the image outwards.
    # W starts as AN.
    # Iteration i=N: we append HN, VN, DN around W.
    # Iteration i=N-1: ...
    
    # The loop `for i = size(s,1)-2 : -1 : 1` in MATLAB actually means:
    # `size(S,1)-2` is index of Coarsest Detail Size (Sz2 in example).
    # Wait. S structure:
    # row 0: Sz A_N
    # row 1: Sz Det_N (Same as A_N)
    # row 2: Sz Det_N-1
    # ...
    # row N: Sz Det_1
    # row N+1: Sz Image
    
    # Logic matches C layout.
    
    for i in range(N_levels):
        # Current Level (starts at N, goes to 1).
        # In this loop structure, we just consume C linearly.
        
        # Detail Size for this level
        # S index: i corresponds to Level index?
        # Level N corresponds to S[1].
        # Level N-k corresponds to S[1+k].
        level_sz = s[i+1] # Row 1, 2, ...
        pixels = level_sz[0] * level_sz[1]
        
        # Extract H, V, D
        h = c[current_idx_c : current_idx_c+pixels].reshape(level_sz)
        current_idx_c += pixels
        
        v = c[current_idx_c : current_idx_c+pixels].reshape(level_sz)
        current_idx_c += pixels
        
        d = c[current_idx_c : current_idx_c+pixels].reshape(level_sz)
        current_idx_c += pixels
        
        # Normalize H, V, D
        if absflag:
            h = mat2gray(np.abs(h), [0, cdx])
            v = mat2gray(np.abs(v), [0, cdx])
            d = mat2gray(np.abs(d), [0, cdx])
        else:
            h = mat2gray(h, [-cdx, cdx])
            v = mat2gray(v, [-cdx, cdx])
            d = mat2gray(d, [-cdx, cdx])
            
        # Pad logic
        # Pad H, V, D to match size of W (the accumulated image so far)
        # Usually W size matches level_sz if dimensions are nice powers of 2.
        # But if not, padding handles mismatch.
        # "ws = size(w)"
        # "pad = ws - size(h)"
        
        ws = w.shape
        pad_h = np.array(ws) - np.array(h.shape)
        
        # Padding function
        def pad_coeff(arr, p_total, val):
            if np.all(p_total == 0): return arr
            p_pre = np.round(p_total / 2).astype(int)
            p_post = p_total - p_pre
            # np.pad expects ((top, bot), (left, right))
            return np.pad(arr, ((p_pre[0], p_post[0]), (p_pre[1], p_post[1])), 
                          mode='constant', constant_values=val)

        h = pad_coeff(h, pad_h, fill)
        v = pad_coeff(v, pad_h, fill)
        d = pad_coeff(d, pad_h, fill)
        
        # Border
        # 'append' adds row/col. 'absorb' replaces last row/col.
        
        if border == 'append':
             # Pad w, h, v
             w = np.pad(w, ((0,1), (0,1)), mode='constant', constant_values=1)
             h = np.pad(h, ((0,1), (0,0)), mode='constant', constant_values=1)
             v = np.pad(v, ((0,0), (0,1)), mode='constant', constant_values=1)
             d = d # d isn't padded? Wait.
             # MATLAB: w=pad(post 1,1), h=pad(post 1,0), v=pad(post 0,1).
             # It expands them to make space for a white line between blocks.
             # D is the corner, doesn't need padding?
             # Concatenation: [W H; V D].
             # If W is (M+1, N+1), H is (M+1, N)??
             # Let's check dims.
             # Initial: W(M,N). H(M,N).
             # After append pad: W(M+1, N+1). H(M+1, N).
             # V(M, N+1). D(M,N).
             # Concat: [W H]. Height match? W is M+1. H is M+1. OK.
             # [V D]. Height M? No, V is M. D is M. OK.
             # Width? [W H] -> N+1 + N.
             # [V D] -> N+1 + N. OK.
             pass
        elif border == 'absorb':
             w[:, -1] = 1; w[-1, :] = 1
             h[-1, :] = 1; v[:, -1] = 1
        
        # Concatenate
        # Top = [W, H]
        # Bot = [V, D]
        # Result = [Top; Bot]
        # BUT dimensions must match.
        # If 'append' caused W to grow to M+1, H to M+1...
        # Check H vs D?
        # If H is (M+1, N) and D is (M, N).
        # [W H] is Height M+1.
        # [V D] is Height M.
        # This works.
        
        row1 = np.hstack((w, h))
        row2 = np.hstack((v, d))
        w = np.vstack((row1, row2))
        
    return w

def mat2gray(img, limits=None):
    """
    Converts matrix to grayscale image [0, 1].
    """
    img = np.array(img, dtype=float)
    if limits is None:
        low = np.min(img)
        high = np.max(img)
    else:
        low, high = limits
        
    if high == low:
        # Avoid div zero
        return np.zeros_like(img)
        
    out = (img - low) / (high - low)
    out = np.clip(out, 0, 1)
    return out
