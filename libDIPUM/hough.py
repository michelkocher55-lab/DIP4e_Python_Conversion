
import numpy as np
from scipy.sparse import coo_matrix

def hough(f, dtheta=1, drho=1):
    """
    Compute the Hough transform of a binary image.
    
    Parameters:
        f: Binary image (2D array).
        dtheta: Spacing of theta (degrees). Default 1.
        drho: Spacing of rho (pixels). Default 1.
        
    Returns:
        H: Hough transform accumulator (2D array).
        theta: Theta values in DEGREES.
        rho: Rho values.
    """
    
    f = f > 0 # Ensure binary
    
    # 1. Theta range
    # MATLAB default: -90 to 89 degrees with step 1
    theta_deg = np.arange(-90, 90, dtheta)
    theta_rad = np.deg2rad(theta_deg)
    
    # 2. Rho range
    # MATLAB: D = sqrt((M-1)^2 + (N-1)^2)
    # rho = -D:drho:D
    rows, cols = f.shape
    D = np.sqrt((rows - 1)**2 + (cols - 1)**2)
    q = np.ceil(D / drho)
    nrho = int(2 * q + 1)
    rho = np.linspace(-q * drho, q * drho, nrho)
    
    # 3. Find non-zero points
    # MATLAB: [x, y, val] = find(f) ? 
    # In MATLAB [r, c] = find(f). r is row (y), c is col (x).
    # Formula: rho = x*cos(theta) + y*sin(theta).
    # So x corresponds to column index, y to row index.
    
    # NumPy: y, x = nonzero(f) -> row indices, col indices
    y, x = np.nonzero(f)
    
    # Weight
    # Standard hough takes binary, so weight is 1.
    val = np.ones_like(x)
    
    # 4. Computation
    # Vectorized accumulation
    # Create grids? No, iterate points or broadcast?
    # x is (K,), theta is (T,)
    # rho_val = x * cosT + y * sinT -> (K, T)
    
    # Use broadcasting
    # shape (K, 1) * (1, T) -> (K, T)
    
    x_mat = x[:, np.newaxis]
    y_mat = y[:, np.newaxis]
    theta_mat = theta_rad[np.newaxis, :]
    
    rho_vals = x_mat * np.cos(theta_mat) + y_mat * np.sin(theta_mat)
    
    # Map rho_vals to bin indices
    # index = round(slope * (rho_vals - rho_start))
    # slope = (nrho - 1) / (rho_end - rho_start)
    # rho_start = rho[0]
    
    # Ideally: index = round((rho_vals - rho[0]) / drho)
    # 0-based index.
    
    rho_start = rho[0]
    # To match MATLAB round behavior (round half up or to nearest even? MATLAB rounds to nearest, half away from zero usually)
    # np.round matches nearest even for .5. 
    # MATLAB's round(X) rounds to nearest integer.
    # Let's simple arithmetic: idx = floor(vals / drho + 0.5)? 
    
    idx = np.round((rho_vals - rho_start) / drho).astype(int)
    
    # Clip to valid range [0, nrho-1] just in case of float errors
    idx = np.clip(idx, 0, nrho - 1)
    
    # Accumulate
    # We have (row, col) pairs in the accumulator plane: (rho_idx, theta_idx)
    # (K, T) indices.
    # Flatten
    
    theta_indices = np.arange(len(theta_deg))
    theta_idx_mat = np.tile(theta_indices, (len(x), 1))
    
    flat_rho_idx = idx.flatten()
    flat_theta_idx = theta_idx_mat.flatten()
    flat_weights = np.tile(val[:, np.newaxis], (1, len(theta_deg))).flatten()
    
    # Accumulate using bincount or sparse
    # H size (nrho, ntheta)
    
    # Using coo_matrix to accumulate
    H_sparse = coo_matrix((flat_weights, (flat_rho_idx, flat_theta_idx)), shape=(nrho, len(theta_deg)))
    H = H_sparse.toarray()
    
    return H, theta_deg, rho
