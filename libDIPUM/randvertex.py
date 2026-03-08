import numpy as np

def randvertex(x, y, npix):
    """
    Adds random noise to the vertices of a polygon.
    
    [xn, yn] = randvertex(x, y, npix) adds uniformly distributed noise to
    the coordinates of vertices of a polygon.
    
    Parameters
    ----------
    x, y : array_like
        Coordinates of vertices.
    npix : float or int
        Maximum number of pixel locations to deviate.
        
    Returns
    -------
    xn, yn : numpy.ndarray
        New coordinates with noise added (rounded and clamped to >= 0).
    """
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    L = len(x)
    if len(y) != L:
        raise ValueError("x and y must have same length")
        
    # Generate noise
    # MATLAB: rand(L, 1) -> uniform [0, 1]
    xnoise = np.random.rand(L)
    ynoise = np.random.rand(L)
    
    # Calculate deviation matches MATLAB logic exactly
    # dev = npix * noise * sign(noise - 0.5)
    # Note: This creates a distribution with a gap in the center [-0.5*npix, 0.5*npix]?
    # Let's verify:
    # If noise = 0.4 -> sign(-0.1)=-1 -> dev = npix * 0.4 * -1 = -0.4 npix.
    # If noise = 0.1 -> sign(-0.4)=-1 -> dev = npix * 0.1 * -1 = -0.1 npix.
    # If noise = 0.5 -> sign(0)=0 -> dev = 0.
    # If noise = 0.6 -> sign(0.1)=1 -> dev = npix * 0.6 * 1 = 0.6 npix.
    # Ranges: [-npix, -0.5*npix] U [0.5*npix, npix] ??
    # Only if noise > 0.5.
    # 0.0 ... 0.5 -> dev in [0, -0.5*npix] -> [-0.5*npix, 0].
    # 0.5 ... 1.0 -> dev in [0.5*npix, npix].
    # So the range is [-0.5*npix, 0] and [0.5*npix, npix].
    # It seems to avoid small deviations around 0? No, it avoids deviations in (0, 0.5*npix) and (-0.5*npix, 0)?
    # Wait, 0.1 noise gives -0.1 npix. That IS small deviation.
    # The gap is in the *middle* of the possible *magnitudes*?
    # No, gap is [0, 0.5*npix] in positive band?
    # noise [0.5, 1.0] -> dev [0.5*npix, 1.0*npix].
    # noise [0.0, 0.5] -> dev [0.0, -0.5*npix].
    # So effectively deviations are in [-0.5*npix, 0] or [0.5*npix, npix].
    # It *excludes* positive deviations between 0 and 0.5*npix.
    # And it *excludes* negative deviations between -npix and -0.5*npix.
    # Okay, weird distribution, but I will implement it faithfully.
    
    xdev = npix * xnoise * np.sign(xnoise - 0.5)
    ydev = npix * ynoise * np.sign(ynoise - 0.5)
    
    # Add noise and round
    xn = np.round(x + xdev)
    yn = np.round(y + ydev)
    
    # Clamp to >= 0 (Python convention, MATLAB was 1)
    xn = np.maximum(xn, 0)
    yn = np.maximum(yn, 0)
    
    return xn.astype(int), yn.astype(int)
