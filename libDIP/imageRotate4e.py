import numpy as np

def imageRotate4e(f, theta, mode='crop'):
    """
    Rotates an image by a specified angle.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image (2D or 3D).
    theta : float
        Angle in degrees. Positive is counter-clockwise.
    mode : str, optional
        'crop' (default) or 'full'.
        
    Returns:
    --------
    gout : numpy.ndarray
        Rotated image.
    """
    f = np.array(f)
    
    # Image size
    M, N = f.shape[:2]
    
    # Make f odd dimensions for symmetric rotation center
    # MATLAB:
    # if iseven(M): f(M+1,:) = f(M,:); oddflagx = true;
    # if iseven(N): f(:,N+1) = f(:,N); oddflagy = true;
    
    oddflagx = False
    oddflagy = False
    
    f_odd = f
    
    if M % 2 == 0:
        # Replicate last row
        # np.pad can do this. pad width (0, 1) for axis 0.
        # But for 3D array we need care.
        if f.ndim == 2:
            f_odd = np.pad(f_odd, ((0, 1), (0, 0)), mode='edge')
        elif f.ndim == 3:
            f_odd = np.pad(f_odd, ((0, 1), (0, 0), (0, 0)), mode='edge')
        oddflagx = True
        
    if N % 2 == 0:
        # Replicate last col
        if f.ndim == 2:
            f_odd = np.pad(f_odd, ((0, 0), (0, 1)), mode='edge')
        elif f.ndim == 3:
            f_odd = np.pad(f_odd, ((0, 0), (0, 1), (0, 0)), mode='edge')
        oddflagy = True
        
    Modd, Nodd = f_odd.shape[:2]
    
    # Convert angle to radians
    theta_rad = np.radians(theta)
    
    # Coordinates of center of input image
    # MATLAB: cx = floor(Modd/2) + 1. (1-based)
    # Python 0-based: index of center is floor(Modd/2).
    # Wait, e.g. M=3. floor(1.5)=1. Indices 0, 1, 2. Center is 1.
    # MATLAB: cx=2.
    # Coordinates logic depends on how we build the coordinate system.
    # MATLAB assigns coordinates 1..M.
    # Python we assign 0..M-1.
    # If we shift by center, we subtract cx.
    # MATLAB: u = 1:M; u = u - cx;
    # Python: u = 0:M-1; u = u - cx_py?
    # cx_py = Modd // 2.
    # If Modd=3, cx=1. u=[0,1,2]. u-cx = [-1, 0, 1]. Symmetric.
    # Matches physical location.
    
    cx = Modd // 2
    cy = Nodd // 2
    
    # Output dimensions
    # Mout = ceil(sqrt(Modd^2 + Nodd^2))
    # Make odd
    diag = np.sqrt(Modd**2 + Nodd**2)
    Mout = int(np.ceil(diag))
    if Mout % 2 == 0:
        Mout += 1
    Nout = Mout
    
    cxp = Mout // 2
    cyp = Nout // 2
    
    # Rotation Matrix (Inverse)
    # MATLAB uses A for mapping FROM output TO input.
    # A = [cos sin (-cxp*cos - cyp*sin); -sin cos (cxp*sin - cyp*cos); 0 0 1]
    # Note: MATLAB [y, x] = meshgrid(...)
    # x is columns (axis 1), y is rows (axis 0).
    # But wait, imageRotate4e comments:
    # "meshgrid uses (col,row) format... we call them [xp,yp]"
    # "[yp, xp] = meshgrid(1:Nout, 1:Mout)"
    # xp corresponds to Mout (rows), yp corresponds to Nout (cols).
    # Wait.
    # MATLAB `meshgrid(1:N, 1:M)` -> returns X (M x N), Y (M x N).
    # X changes along columns (1..N). Y changes along rows (1..M).
    # So Y corresponds to ROWS. X corresponds to COLS.
    # In `imageRotate4e.m`: `[yp, xp] = meshgrid(1:Nout, 1:Mout)`.
    # yp takes values from 1:Nout (Cols).
    # xp takes values from 1:Mout (Rows).
    # So xp is ROWS, yp is COLS.
    
    # Python:
    # rows = 0..Mout-1. cols = 0..Nout-1.
    # xp, yp = np.meshgrid(rows, cols, indexing='ij')
    # xp (Mout, Nout) varies along axis 0.
    # yp (Mout, Nout) varies along axis 1.
    # So xp matches rows, yp matches cols.
    
    # Coordinates adjustment
    # We want Euclidean coordinates relative to center?
    # No, the logic constructs coordinates 1..Mout (indices) and maps them to indices 1..M.
    # The A transform handles the shift.
    # Matrix A in MATLAB logic operates on [xp; yp; 1].
    # where xp, yp are indices.
    # It maps them to [orig_r; orig_c; 1].
    # Then `orig_r = orig_r + cx`, `orig_c = orig_c + cy`.
    # Actually line 95: `origCoords(1,:) = origCoords(1,:) + cx`.
    # And A has translation terms involving -cxp, -cyp.
    # Basically:
    # Input System: Center at (cx, cy).
    # Output System: Center at (cxp, cyp).
    # We map Output Indices (xp, yp) -> Centered Output Coords (xp-cxp, yp-cyp)
    # -> Rotate -> Centered Input Coords (u, v)
    # -> Input Indices (u+cx, v+cy).
    
    # Let's verify A:
    # [ u ]   [ c  s ] [ xp - cxp ]
    # [ v ] = [ -s c ] [ yp - cyp ]
    # u = c(xp-cxp) + s(yp-cyp) = c*xp + s*yp - (c*cxp + s*cyp)
    # v = -s(xp-cxp) + c(yp-cyp) = -s*xp + c*yp - (-s*cxp + c*cyp) = -s*xp + c*yp + (s*cxp - c*cyp)
    # Comparison with MATLAB A:
    # Row 1: [cos sin (-cxp*cos - cyp*sin)] -> c*xp + s*yp - ... MATCHES.
    # Row 2: [-sin cos (cxp*sin - cyp*cos)] -> -s*xp + c*yp + ... MATCHES.
    
    # Python Implementation:
    # We use 0-based indices for xp, yp, cx, cy, cxp, cyp.
    # The math is identical because shifts are relative.
    
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    # Note: cxp, cyp are scalars (centers).
    # Matrix multiplication logic for vectorized:
    # Coords matrix [3, N_pixels].
    
    xp_vals = np.arange(Mout)
    yp_vals = np.arange(Nout)
    xp_grid, yp_grid = np.meshgrid(xp_vals, yp_vals, indexing='ij')
    
    # Flatten
    primeCoords = np.stack([xp_grid.ravel(), yp_grid.ravel(), np.ones_like(xp_grid.ravel())])
    
    # Matrix A
    # Row 1: c, s, offset_x
    # Row 2: -s, c, offset_y
    
    t_x = -cxp * cos_t - cyp * sin_t
    t_y = cxp * sin_t - cyp * cos_t
    
    A = np.array([
        [cos_t, sin_t, t_x],
        [-sin_t, cos_t, t_y],
        [0, 0, 1]
    ])
    
    # Map
    origCoords = A @ primeCoords
    
    # Round nearest neighbor
    origCoords = np.round(origCoords).astype(int)
    
    # Uncenter
    # Add cx, cy to mapped coords
    origCoords[0, :] += cx
    origCoords[1, :] += cy
    
    # Valid mask
    # 0 <= r < Modd, 0 <= c < Nodd
    r_coords = origCoords[0, :]
    c_coords = origCoords[1, :]
    
    valid_mask = (r_coords >= 0) & (r_coords < Modd) & \
                 (c_coords >= 0) & (c_coords < Nodd)
                 
    # Map pixels
    # We need to map linear indices or use fancy indexing
    
    # Initialize output
    # Handle multi-channel
    if f.ndim == 3:
        n_chan = f.shape[2]
        g = np.zeros((Mout, Nout, n_chan), dtype=f.dtype)
        
        # Valid indices in output (where mask is true)
        # primeCoords correspond to flattening of g.
        # But we need (r,c) of output.
        # xp_grid.ravel()[valid_mask] gives rows.
        out_r = xp_grid.ravel()[valid_mask]
        out_c = yp_grid.ravel()[valid_mask]
        
        # Original coords
        in_r = r_coords[valid_mask]
        in_c = c_coords[valid_mask]
        
        # Assignment
        g[out_r, out_c, :] = f_odd[in_r, in_c, :]
        
    else:
        g = np.zeros((Mout, Nout), dtype=f.dtype)
        out_r = xp_grid.ravel()[valid_mask]
        out_c = yp_grid.ravel()[valid_mask]
        in_r = r_coords[valid_mask]
        in_c = c_coords[valid_mask]
        g[out_r, out_c] = f_odd[in_r, in_c]
        
    # Mode handling
    if mode == 'crop':
        # Extract Modd x Nodd from center of g
        # extractImage logic in MATLAB:
        # cgx = floor(Mg/2)+1...
        # Basically center crop.
        
        cgx = Mout // 2
        cgy = Nout // 2
        
        start_x = cgx - (Modd - 1) // 2
        start_y = cgy - (Nodd - 1) // 2
        
        end_x = start_x + Modd
        end_y = start_y + Nodd
        
        # Python slicing is exclusive at end
        gext = g[int(start_x):int(end_x), int(start_y):int(end_y)]
        
        # Remove padding if added
        if oddflagx:
            gext = gext[:-1, ...]
        if oddflagy:
            gext = gext[:, :-1, ...]
            
        return gext
        
    elif mode == 'full':
        return g
    
    else:
        raise ValueError("Unknown mode: " + mode)
