import numpy as np

def laplacianTF4e(P, Q):
    """
    Laplacian transfer function in the frequency domain.
    
    Parameters:
    -----------
    P : int
        Number of rows (padded size).
    Q : int
        Number of columns (padded size).
        
    Returns:
    --------
    H : numpy.ndarray
        Laplacian transfer function (P x Q).
        Real and negative values.
    """
    
    # MATLAB: Ucenter = floor(P/2) + 1;
    # (1-based index).
    # In 0-based indexing:
    # floor(P/2) is the index of the center.
    
    # MATLAB: U = 0:P-1; U = U - Ucenter;
    # U ranges from -floor(P/2) to P-1-floor(P/2).
    # Example P=4. Ucenter=3 (1-based) -> 2 (0-based?).
    # Wait, MATLAB: floor(4/2)+1 = 3. 
    # U=[1,2,3,4] (1-based indices) -> mapped to distances?
    # Source Code: U = 0:P-1. U = U - Ucenter.
    # P=4. U=[0, 1, 2, 3]. Ucenter=3.
    # U-Ucenter = [-3, -2, -1, 0]. This puts the DC at the end?
    # Usually we want DC in the center for visualization if centered, 
    # or at (0,0) for FFT.
    # The comments say "H is centered in the P x Q frequency rectangle."
    # So we want the center of the array to be frequency 0.
    
    # Let's verify MATLAB behavior for P=4.
    # Ucenter = 3.
    # U = [-3, -2, -1, 0].
    # This seems off for a centered representation.
    # Usually centered means symmetric around Ucenter.
    
    # Re-reading:
    # "Ucenter = floor(P/2) + 1"
    # "U = U - Ucenter"
    # If P is even (4), range is [-3, 0]. Not symmetric.
    # If P is odd (5), Ucenter=3. U=[0,1,2,3,4]. U-3=[-3,-2,-1,0,1]. Not symmetric.
    
    # Maybe standard centered convention:
    # ceil((P-1)/2)?
    # Or maybe the source wants it shifted?
    # "H is centered...".
    
    # Let's replicate strict logic of the provided MATLAB code.
    # Ucenter = floor(P/2) + 1. (In MATLAB 1-based logic).
    # But U = 0:P-1 creates a 0-based vector.
    # So `U - Ucenter` subtracts the scalar.
    
    # Equivalent Python:
    # P=4. Ucenter_matlab = 3.
    # U_py = [0, 1, 2, 3].
    # Result = [-3, -2, -1, 0].
    
    # Is this what we want?
    # Laplacian filter H = -4*pi^2*(u^2+v^2).
    # If u=0 is at end, then DC is at end corner?
    # This implies the filter is NOT centered in the visual sense (P/2, Q/2), 
    # but maybe aligned such that `fftshift` puts it in valid position?
    # No, typically "centered" means DC is at (P/2, Q/2).
    
    # Check `lpFilterTF4e` implementation:
    # It used `x - P/2`.
    # P=4. x=[0,1,2,3]. Center=2.0.
    # x-P/2 = [-2, -1, 0, 1]. Symmetric-ish? -2^2=4, 1^2=1. Not exactly symmetric.
    
    # Let's look at `laplacianTF4e` MATLAB again.
    # `U = 0:P-1`. `Ucenter = floor(P/2)+1`.
    # `U = U - Ucenter`.
    
    # If P=5. Ucenter = 3.
    # U = [0,1,2,3,4] - 3 = [-3, -2, -1, 0, 1].
    # Squaring: [9, 4, 1, 0, 1].
    # Center (0) is at index 3 (which is 4th element).
    # In 1-based indexing, center is at 3.
    # In 0-based indexing U[3] is 0.
    # So DC component is at index 3.
    
    # If P=256 (even). Ucenter = 129.
    # U = 0..255.
    # U - 129 = -129 ... 126.
    # DC (0) is at index 129 (130th element).
    
    # This seems to shift the DC component to roughly the center.
    # I will replicate this exact logic.
    
    u_center = np.floor(P/2) + 1
    v_center = np.floor(Q/2) + 1
    
    rows = np.arange(P)
    cols = np.arange(Q)
    
    u = rows - u_center
    v = cols - v_center
    
    # Coordinates [v, u] = meshgrid(V, U)
    # MATLAB: U (rows?), V (cols?).
    # "MATLAB uses opposite convention ... hence reversed u and v ... origin at top left".
    # U is associated with rows (top-down), V with cols (left-right).
    
    # Python meshgrid(rows, cols, indexing='ij') returns u_grid (P, Q), v_grid (P, Q).
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    
    # H = -4 * pi^2 * (u^2 + v^2)
    H = -4 * (np.pi**2) * (u_grid**2 + v_grid**2)
    
    return H
