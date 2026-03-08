
import numpy as np
import math

def edgemodel(edge_type, M, N, ILOW, IHIGH, WIDTH):
    """
    Generates an image containing an edge of specified model.
    
    Parameters:
    edge_type : str
        'step', 'ramp', or 'roof'.
    M, N : int
        Size of the output image (M rows, N columns).
    ILOW : float
        Intensity on the left side (and right for roof).
    IHIGH : float
        Intensity on the right side (or peak for roof).
    WIDTH : int
        Width of the ramp or roof edge.
        
    Returns:
    f : ndarray
        MxN image containing the vertical edge profile.
    """
    
    edge_type = edge_type.lower()
    
    # Initialize profile
    # MATLAB: 1-based indexing.
    # midpoint = floor(N/2);
    # profile(1:midpoint) = ILOW;
    # profile(midpoint+1:N) = IHIGH;
    
    profile = np.zeros(N, dtype=float)
    midpoint = int(np.floor(N / 2)) # Python 0-based index of the start of the right side?
    # MATLAB: 1..midpoint is left. midpoint+1..N is right.
    # Python: 0..midpoint-1 is left. midpoint..N-1 is right?
    # Let's align with MATLAB logic.
    # If N=10, midpoint=5.
    # MATLAB: 1..5 (5 pixels) ILOW. 6..10 (5 pixels) IHIGH.
    # Python: 0..5 (5 pixels) ILOW -> profile[:midpoint]
    #         5..10 (5 pixels) IHIGH -> profile[midpoint:]
    
    profile[:midpoint] = ILOW
    profile[midpoint:] = IHIGH
    
    # Compute first and last point of edges.
    if edge_type == 'roof' and (WIDTH % 2) == 0:
        WIDTH += 1 # Make WIDTH odd so roof edge will be symmetrical.
        
    # MATLAB: firstpoint = midpoint - floor(WIDTH/2);
    # In MATLAB indices.
    # Python indices should be consistent if we strictly map.
    # firstpoint_idx = midpoint - floor(WIDTH/2) - 1 ? No, let's keep Python 0-based logic.
    # If midpoint is index 5 (6th pixel).
    # WIDTH=3. floor(3/2)=1.
    # firstpoint = 5 - 1 = 4.
    # lastpoint = 4 + 3 - 1 = 6.
    # Pts: 4, 5, 6. (3 pixels). Centered at 5? Yes.
    
    firstpoint = midpoint - int(np.floor(WIDTH / 2))
    lastpoint = firstpoint + WIDTH - 1
    
    # Modify profile based on type
    if edge_type == 'step':
        pass
        
    elif edge_type == 'ramp':
        if WIDTH == 1:
            pass
        else:
            # DEL = (IHIGH - ILOW)/(WIDTH - 1);
            DEL = (IHIGH - ILOW) / (WIDTH - 1)
            
            # profile(firstpoint) = ILOW; (Already set mostly)
            # profile(lastpoint) = IHIGH; (Already set)
            profile[firstpoint] = ILOW
            profile[lastpoint] = IHIGH
            
            # Loop
            # MATLAB: for k = firstpoint + 1 : firstpoint + WIDTH - 2
            # Which is indices firstpoint+1 to lastpoint-1
            count = 0
            for k in range(firstpoint + 1, lastpoint): # range excludes end, so lastpoint is excluded
                count += 1
                profile[k] = ILOW + count * DEL
                
    elif edge_type == 'roof':
        profile[:] = ILOW
        
        if WIDTH == 1 or WIDTH == 3:
            profile[midpoint] = IHIGH # Peak
        elif WIDTH == 2:
            # Code says if WIDTH is even it adds 1.
            # So WIDTH 2 becomes 3. 
            # But the 'elseif WIDTH == 2' block in MATLAB assumes it WAS 2?
            # Wait, MATLAB L41: if roof & even -> WIDTH++.
            # So WIDTH is never even here in 'roof'.
            # The 'elseif WIDTH == 2' in MATLAB (L72) might be unreachable if line 41 ran?
            # Ah, L41 modifies WIDTH. So L72 checks the MODIFIED WIDTH?
            # If input was 1 -> WIDTH=1.
            # If input was 2 -> WIDTH=3.
            # If input was 3 -> WIDTH=3.
            # So WIDTH=2 case is impossible?
            # Let's checking L41 again. "Make WIDTH odd".
            # So WIDTH will always be odd.
            # MATLAB L69: if WIDTH == 1 | WIDTH == 3.
            # It handles 1 and 3.
            # What if input was 2 -> becomes 3 -> hits L69.
            # So profile[midpoint] = IHIGH.
            pass
        else:
            # WIDTH > 3 (and odd)
            
            # DEL = (IHIGH - ILOW)/(WIDTH - ceil(WIDTH/2));
            # WIDTH - ceil(WIDTH/2) = floor(WIDTH/2) if odd?
            # e.g. 5. ceil(2.5)=3. 5-3=2. floor(2.5)=2. Correct.
            DEL = (IHIGH - ILOW) / (WIDTH - math.ceil(WIDTH / 2))
            
            profile[firstpoint] = ILOW
            profile[midpoint] = IHIGH
            profile[lastpoint] = ILOW
            
            # Left side
            count = 0
            # MATLAB: for k = firstpoint + 1: midpoint - 1
            for k in range(firstpoint + 1, midpoint):
                count += 1
                profile[k] = ILOW + count * DEL
                
            # Right side
            count = 0
            # MATLAB: for k = midpoint + 1:lastpoint - 1
            for k in range(midpoint + 1, lastpoint):
                count += 1
                profile[k] = IHIGH - count * DEL
                
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")
        
    # Replicate to 2D
    # f = repmat(profile, M, 1)
    f = np.tile(profile, (M, 1))
    
    return f
