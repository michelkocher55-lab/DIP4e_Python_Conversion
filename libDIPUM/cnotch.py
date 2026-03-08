
import numpy as np
import sys
from libDIPUM.hpfilter import hpfilter

def cnotch(type_filter, notch, M, N, C, D0, n=1):
    """
    Generates notch filter transfer functions.
    
    H = cnotch(type, notch, M, N, C, D0, n)
    
    Parameters:
    type_filter: 'ideal', 'butterworth', 'gaussian'
    notch: 'reject', 'pass'
    M, N: Size of the transfer function
    C: Kx2 matrix of notch centers (u, v)
    D0: Cutoff frequency (scalar or length K)
    n: Order (default 1)
    
    Returns:
    H: Notch transfer function (uncentered)
    """
    
    # helper for even check
    def iseven(val):
        return val % 2 == 0
    def isodd(val):
        return val % 2 != 0

    # Define largest odd dimensions
    MO = M
    NO = N
    if iseven(M):
        MO = M - 1
    if iseven(N):
        NO = N - 1
        
    # Center of TF (0-based index)
    # MATLAB: floor(MO/2) + 1. (Index of center element)
    # Python: MO//2. (Index of center element)
    # Example MO=5. Center index 2 (element 3). 
    # MATLAB: floor(2.5)+1 = 3. Correct.
    center = np.array([MO // 2, NO // 2])
    
    # Number of notch pairs
    C = np.atleast_2d(C)
    K = C.shape[0]
    
    # D0 handling
    if np.isscalar(D0):
        D0 = np.full(K, D0, dtype=float)
    else:
        D0 = np.array(D0, dtype=float)
        if len(D0) < K:
             # Should match? MATLAB implicitly expands logic manual: D0(1:K) = D0.
             # Only if scalar. If vector provided, assumes correct length?
             pass 

    # Shift notch centers
    # MATLAB: C registers them W.R.T center of rect.
    # C coordinates are given W.R.T standard origin (top-left).
    # To get shift relative to center: C_shifted = C - center.
    # Note: center is (row_center, col_center) = (u_center, v_center).
    # C is (u, v) = (row, col).
    
    center_rep = np.tile(center, (K, 1))
    C_shift = C - center_rep
    
    # Generate reject filter
    H = rejectFilter(type_filter, MO, NO, D0, K, C_shift, n)
    
    # Process output (padding, passing)
    H = processOutput(notch, H, M, N, center)
    
    return H

def rejectFilter(type_filter, MO, NO, D0, K, C, n):
    # Initialize H as all-pass
    H = np.ones((MO, NO), dtype=float)
    
    for I in range(K):
        # Place notch
        # Usize, Vsize: size of the filter to generate
        # C[I, 0] is delu, C[I, 1] is delv
        delu = C[I, 0]
        delv = C[I, 1]
        
        Usize = MO + 2 * int(abs(delu))
        Vsize = NO + 2 * int(abs(delv))
        
        # Determine filter type string for hpfilter
        # Assuming hpfilter is available. 
        # Note: MATLAB `fftshift(hpfilter(...))`
        # hpfilter returns uncentered. fftshift centers it.
        # Python hpfilter needs to match.
        
        # We need to import hpfilter inside or assume global.
        # Assuming it matches signature: hpfilter(type, M, N, D0, n)
        
        filt_uncentered = hpfilter(type_filter, Usize, Vsize, D0[I], n)
        filt = np.fft.fftshift(filt_uncentered)
        
        # Insert
        H = placeNotches(H, filt, delu, delv)
        
    return H

def placeNotches(H, filt, delu, delv):
    M, N = H.shape
    U = 2 * int(abs(delu))
    V = 2 * int(abs(delv))
    
    # Calculate overlap
    # Python slicing: 0-based.
    # filt is larger than H.
    
    # MATLAB:
    # if delu >= 0 && delv >= 0 (Q4 in cartesion/img coords differ? logic says Q1)
    # filtCommon = filt(1:M, 1:N);
    # In Python: filt[0:M, 0:N]
    
    if delu >= 0 and delv >= 0:
        filtCommon = filt[0:M, 0:N]
    elif delu < 0 and delv >= 0:
        # MATLAB: filt(U+1:U+M, 1:N)
        # Python: filt[U:U+M, 0:N]
        filtCommon = filt[U:U+M, 0:N]
    elif delu < 0 and delv < 0:
        # MATLAB: filt(U+1:U+M, V+1:V+N)
        # Python: filt[U:U+M, V:V+N]
        filtCommon = filt[U:U+M, V:V+N]
    elif delu >= 0 and delv <= 0:
        # MATLAB: filt(1:M, V+1:V+N)
        # Python: filt[0:M, V:V+N]
        filtCommon = filt[0:M, V:V+N]
    else:
        # Should not happen
        filtCommon = np.zeros((M, N))

    # Product
    P = np.ones((M, N)) * filtCommon
    
    # Symmetry: Rotate 180 (rot90(P, 2))
    # MATLAB rot90(A, k) rotates counterclockwise.
    # Python np.rot90(m, k) rotates counterclockwise.
    P_sym = P * np.rot90(P, 2)
    
    # Update H: H .* P
    H_out = H * P_sym
    
    return H_out

def processOutput(notch, H, M, N, center):
    # H is odd dims.
    # center is [MO//2, NO//2] (indices).
    # MATLAB `center` variable passed here was `floor(MO/2)+1`.
    # Let's verify `center` passed.
    # In main cnotch, `center` is `np.array([MO // 2, NO // 2])`.
    
    centerU = center[0]
    centerV = center[1]
    
    # Expand if even
    # Duplicating first row/col and making symmetric.
    # Note: MATLAB `H(1, :)`. In Python `H[0, :]`.
    
    # Helper to flip part
    def make_symmetric_row(row, c):
        # row is 1D array.
        # c is center index of the *source* (odd array).
        # MATLAB: newRow(1:centerV-1) = fliplr(newRow(centerV+1:end))
        # Python: newRow[0:c] = newRow[c+1:][::-1]
        # Check lengths.
        # Arr len L (odd). c = L//2.
        # 0..c-1 has length c.
        # c+1..end has length L - (c+1) = (2c+1) - c - 1 = c.
        # Matches.
        new_row = row.copy()
        new_row[0:c] = new_row[c+1:][::-1]
        return new_row

    newRow = make_symmetric_row(H[0, :], centerV)
    newCol = make_symmetric_row(H[:, 0], centerU)
    
    Hout = H
    
    iseven_M = (M % 2 == 0)
    iseven_N = (N % 2 == 0)
    isodd_M = not iseven_M
    isodd_N = not iseven_N
    
    if iseven_M and iseven_N:
        # MATLAB: cat(1, newRow, H) -> stack vertically
        Hout = np.vstack((newRow, Hout))
        # MATLAB: newCol = cat(1, H(1,1), newCol) -> wait.
        # MATLAB logic: newCol for the final output needs to match height?
        # No, newCol is derived from H(:,1). Length MO.
        # After vstack, Hout has height MO+1 = M.
        # The new column needs to be length M?
        # MATLAB: `newCol = cat(1,H(1,1),newCol);`
        # H(1,1) is the corner.
        # Then `Hout = cat(2,newCol,Hout);`
        # So newCol is prepended to the left.
        
        # Logic check:
        # newCol original len MO.
        # We need len MO+1.
        # We prepend H[0,0] (the corner value of original H? or the new Hout?)
        # MATLAB code: `newCol = cat(1, H(1,1), newCol)`. This uses H(1,1) from *original* H (passed as arg).
        # Yes, H is arg.
        # But wait, `Hout` has `newRow` at top. `H(1,1)` is just one pixel.
        # Python:
        col_to_add = np.concatenate(([H[0,0]], newCol))
        col_to_add = col_to_add[:, np.newaxis] # make 2D col
        Hout = np.hstack((col_to_add, Hout))
        
    elif iseven_M and isodd_N:
        Hout = np.vstack((newRow, Hout))
    elif isodd_M and iseven_N:
        # MATLAB: Hout = cat(2, newCol, H)
        # newCol is len MO (matches H).
        col_to_add = newCol[:, np.newaxis]
        Hout = np.hstack((col_to_add, H))
    else:
        Hout = H
        
    # Uncenter
    Hout = np.fft.ifftshift(Hout)
    
    # Notch pass ?
    if notch == 'pass':
        Hout = 1 - Hout
        
    return Hout
