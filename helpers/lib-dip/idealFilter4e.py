import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.transform4e import transform4e
from lib.invTransform4e import invTransform4e
from lib.lpFilterTF4e import lpFilterTF4e
from hpFilterTF4e import hpFilterTF4e # We assume hp filter wrapper exists or we use 1-lp

def idealFilter4e(f, xform, type, r):
    """
    Ideal filter 2-D input F using XFORM transform and filter of radius R.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input square image.
    xform : str
        Transform name (e.g., 'DFT', 'DCT').
    type : str
        'LP' or 'HP'.
    r : float
        Radius/Cutoff.
        
    Returns:
    --------
    g : numpy.ndarray
        Filtered image.
    """
    f = np.array(f, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
         raise ValueError("Input F must be a square matrix!")
         
    n = f.shape[0]
    
    # Zero pad the input to 2N x 2N
    # MATLAB: padarray(f, [n n], 0, 'post') -> Pads [0..n-1] with 0s to end?
    # padarray(A, [r c], val, 'post') appends r rows and c cols.
    # So size becomes n+n = 2n.
    
    # F = np.pad(f, ((0, n), (0, n)), mode='constant')
    F = np.pad(f, ((0, n), (0, n)), 'constant')
    
    # Compute transform
    # Note: idealFilter4e uses transform4e on 2N x 2N padded image
    # Note: Our transform4e handles square matrices, so it's fine.
    T = transform4e(F, xform)
    
    # Filter Creation
    # Check if DCT/DST (Special handling in MATLAB because of resolution?)
    # MATLAB: "diameter of the filter is doubled to account for the 2x frequency resolution of the sin and cosine transforms"
    
    is_dct_dst = xform.upper() in ['DCT', 'DST']
    
    if is_dct_dst:
        # Create mask manually
        # H = zeros(2*n, 2*n)
        # distance <= 2*r
        # Loop based creation in MATLAB.
        # Vectorized here:
        
        rows = np.arange(2 * n)
        cols = np.arange(2 * n)
        r_grid, c_grid = np.meshgrid(rows, cols, indexing='ij')
        # Dist from (0,0)? MATLAB: (i-1)^2 + (j-1)^2 <= (2*r)^2
        # Yes, from origin.
        
        dist = np.sqrt(r_grid**2 + c_grid**2)
        H = np.zeros((2 * n, 2 * n))
        H[dist <= 2 * r] = 1.0
        
        if type.upper() == 'HP':
            H = 1.0 - H
            
    else:
        # DFT / DHT
        # Use lpfilter/hpfilter (centered)
        # MATLAB calls lpfilter('ideal', 2n, 2n, r)
        # Assuming our lpFilterTF4e returns centered filter
        
        # NOTE: transform4e uses tmat4e which generates Standard Basis.
        # For DFT, tmat4e generates standard DFT matrix.
        # transform4e returns T = A F A.T (Standard DFT, with DC at (0,0) usually?)
        # A = dftmtx.
        # dftmtx order: DC at index 0.
        # So T has DC at corners (0,0).
        # BUT lpFilterTF4e typically generates CENTERED filter (DC at N/2, N/2).
        # We must shift the filter if T is unshifted, or shift T.
        # MATLAB idealFilter4e uses `lpfilter`. Does MATLAB's `lpfilter` return unshifted?
        # Our `lpFilterTF4e.py` returns Centered filter (mesh centered at P/2, Q/2).
        # MATLAB code note in `idealFilter4e.m`:
        # Line 41: "Filter and spectrum display for DFT and DHT"
        # Since T is computed via matrix mul, it is NOT shifted.
        # If we multiply by centered H, we mask wrong frequencies.
        # We need to fftshift the H to corners (unshift it) before multiply.
        
        if type.upper() == 'LP':
             H_centered = lpFilterTF4e('ideal', 2*n, 2*n, r)
        elif type.upper() == 'HP':
             # H_centered = hpFilterTF4e('ideal', 2*n, 2*n, r)
             # Use 1 - LP to be safe if hp wrapper doesn't exist
             # But we implemented hpFilterTF4e.
             from hpFilterTF4e import hpFilterTF4e
             H_centered = hpFilterTF4e('ideal', 2*n, 2*n, r)
        else:
             raise ValueError("Type must be LP or HP")
             
        # Shift H to corners to match T (which is unshifted)
        # Use ifftshift to move Center to (0,0)
        from scipy.fft import ifftshift
        H = ifftshift(H_centered)
        
    # Apply
    T_filtered = T * H
    
    # Inverse Transform
    G_padded = invTransform4e(T_filtered, xform)
    
    # Crop (MATLAB: g(1:688, 1:688) ... Wait, MATLAB hardcoded 1:688 ????)
    # Oh, check Line 55 in idealFilter4e.m:
    # 55: g = G(1:688, 1:688);
    # That looks like a bug or specific project code in MATLAB source!
    # "Get the size of the input... n = s(1)".
    # It padded to 2n.
    # It should crop back to n.
    # I will implement generically: G[0:n, 0:n].
    
    G = np.real(G_padded) # Take real part
    g = G[:n, :n]
    
    # Normalize/Scale if needed?
    # MATLAB: imshow(mat2gray(abs(g)))
    # It returns 'g' but displays normalized.
    # I will return raw g (or abs(g)?). Return g as is.
    
    return g
