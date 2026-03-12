import numpy as np
import matplotlib.pyplot as plt
from helpers.libdip.tmat4e import tmat4e
from helpers.libdip.intScaling4e import intScaling4e

def basisImage4e(xform='DCT', N=4, P=2, display=True):
    """
    Create the 2-d basis images of transform XFORM of size N x N.

    Parameters:
    -----------
    xform : str, optional
        Transform type. 'DCT', 'DFT', 'DFTr', 'DFTi', 'WHT', 'SLT', 'HAAR', etc.
        Default is 'DCT'.
    N : int, optional
        Size of the basis images. Default is 4. Must be >= 2.
    P : int, optional
        Spacing between basis images. Default is 2. Must be >= 0.
    display : bool, optional
        If True, display the result using matplotlib. Default is True.

    Returns:
    --------
    S_COMPOSITE : numpy.ndarray
        The composite image containing all basis images (no borders/spacing).
    S_DISPLAY : numpy.ndarray
        The composite image containing all basis images with borders and spacing.
    """
    if N < 2:
        print('N must be at least 2!')
        return None
    if P < 0:
        print('P must be at least 0!')
        return None

    # Get transform matrix
    # Handle specialized xform strings for DFT components
    if xform in ['DFTr', 'DFTi']:
        A = tmat4e('DFT', N)
    else:
        A = tmat4e(xform, N)
        
    if A is None:
        return None

    # Defaults for appearance
    space = 1.0
    outline = 0.0

    if xform in ['WHT', 'STD']:
        space = 1.0
        outline = 0.5

    # Composite image size W x W. 
    # N basis functions per dim. Each is size N x N.
    # Total size N*N x N*N
    w = N * N
    S_COMPOSITE = np.zeros((w, w))

    # Generate basis images
    # S = A[i,:].T * A[j,:]
    
    # Python 0-indexed.
    # MATLAB loop i=1:N (rows of A), j=1:N (rows of A)
    # Note: MATLAB basis construction: 
    # S = transpose(A(i,:)) * A(j,:) 
    # Python: A[i, :].reshape(-1, 1) @ A[j, :].reshape(1, -1) -> this creates N x N matrix.
    
    for i in range(N):
        for j in range(N):
            # Outer product of i-th row and j-th row?
            # MATLAB: A(i,:) is 1xN. Transpose is Nx1. A(j,:) is 1xN.
            # product is NxN.
            
            # Complex handling
            if xform == 'DFT':
                # Magnitude
                term = np.outer(A[i, :], A[j, :]) # This creates complex matrix if A is complex
                S = np.abs(term)
            elif xform == 'DFTr':
                term = np.outer(A[i, :], A[j, :])
                S = np.real(term)
            elif xform == 'DFTi':
                term = np.outer(A[i, :], A[j, :])
                S = np.imag(term)
            else:
                term = np.outer(A[i, :], A[j, :])
                S = term

            # Place in composite
            # MATLAB: x = N*i - N + 1 ...
            # i, j are 0-based in Python loop.
            # x_start = N * i
            # y_start = N * j
            
            x_start = N * i
            y_start = N * j
            
            S_COMPOSITE[x_start : x_start + N, y_start : y_start + N] = S

    # Scale to 0-1 (mat2gray)
    # We can use intScaling4e with mode='full' and type_out='floating'
    # Check if complex (though we took abs/real/imag above, except normal path?)
    # tmat4e returns complex for DFT. If xform='DFT', we took abs. 'DFTr' real.
    # other transforms are real. So S_COMPOSITE should be real.
    
    SC = intScaling4e(S_COMPOSITE, mode='full', type_out='floating')
    
    # Add 1-pixel line (outline) around basis images
    # New size: N*N + 2*N (each basis image gets +2 size? No)
    # MATLAB code:
    # w = N*N + 2*N
    # for i... xd = (N+2)*i - N ...
    # Essentially each N x N basis image is placed into a (N+2)x(N+2) cell, centered?
    # No, let's look at indices.
    # MATLAB: xd = (N+2)*i - N
    # if i=1: xd = N+2 - N = 2. (1-based). So padding 1 pixel top/left?
    # if i=N: xd = (N+2)*N - N = N^2 + 2N - N = N^2 + N.
    # The block size is (N+2).
    # It puts S_COMPOSITE block into center of (N+2) block?
    # S_LINE(xd:xd+N-1, ...) = S
    # xd start at 2. Ends at 2+N-1 = N+1.
    # So pixels 1 (border), 2..N+1 (image), N+2 (border).
    # Yes, it adds 1 pixel border around each image.
    
    total_w_line = N * (N + 2) # N blocks of size N+2
    S_LINE = np.full((total_w_line, total_w_line), outline, dtype=float)
    
    for i in range(N):
        for j in range(N):
            # Source coordinates
            x = N * i
            y = N * j
            
            # Dest coordinates
            # Each block has size N+2. Logic: (N+2)*i + 1 (for 1 pixel border offset)
            xd = (N + 2) * i + 1
            yd = (N + 2) * j + 1
            
            S_LINE[xd : xd + N, yd : yd + N] = SC[x : x + N, y : y + N]

    # Add space of width P
    # w = N*(N+2) + P*(N-1)
    # Each block is (N+2). Between blocks we add P?
    # MATLAB loop:
    # xd = (N+P+2)*i - N - 1 - P
    # This logic is getting complex to port directly by index math.
    # Concept:
    # We have N blocks. Each block is "Bordered Basis Image" (size N+2).
    # We want P pixels spacing between them.
    # Total width = N * (N+2) + (N-1) * P.
    
    block_size = N + 2
    total_w_display = N * block_size + (N - 1) * P
    S_DISPLAY = np.full((total_w_display, total_w_display), space, dtype=float)
    
    # We can iterate and place "Bordered Blocks" from S_LINE into S_DISPLAY
    # But S_LINE is contiguous. We need to grab chunks from S_LINE.
    
    for i in range(N):
        for j in range(N):
            # Source (S_LINE) coordinates for the (i,j) block (which is size N+2)
            # block start in S_LINE was (N+2)*i
            xs = block_size * i
            ys = block_size * j
            
            # Dest (S_DISPLAY) coordinates
            # We have i blocks before us, and i spacers of size P? 
            # Wait, spacing is between blocks. i=0 -> 0 spacing. i=1 -> 1 spacing.
            xd = i * (block_size + P)
            yd = j * (block_size + P)
            
            S_DISPLAY[xd : xd + block_size, yd : yd + block_size] = \
                S_LINE[xs : xs + block_size, ys : ys + block_size]

    return S_COMPOSITE, S_DISPLAY
