import numpy as np
import sys
import os
from scipy.fftpack import dct

# Ensure imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers.mat2huff import mat2huff
from libDIP.tmat4e import tmat4e

def im2jpeg4e(f, quality=1, bits=8):
    """
    Compresses an image using a JPEG approximation.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image (unsigned integer).
    quality : float, optional
        Quality factor (default 1). Higher means more loss/compression?
        Wait, MATLAB: "Quality determines the amount of information that is lost and compression achieved."
        MATLAB: m * quality. So larger quality -> larger quantization steps -> more compression/loss.
        Standard JPEG quality is 0-100 where 100 is best.
        This function uses a scale factor 'quality'.
    bits : int, optional
        Bits/pixel (default 8).
        
    Returns:
    --------
    c : dict
        Compressed structure:
        c['size']: image size
        c['bits']: bits
        c['numblocks']: number of blocks
        c['quality']: quality * 100
        c['huffman']: huffman struct from mat2huff
    """
    f = np.array(f)
    
    # Input checks
    if bits < 0 or bits > 16:
        raise ValueError('The input image must have 1 to 16 bits/pixel.')
    if quality <= 0:
        raise ValueError('Input parameter QUALITY must be greater than zero.')
    
    # Check if integer
    if not np.issubdtype(f.dtype, np.integer):
         pass # Allow float but warn? MATLAB errors if not integer.
         # We'll cast to float for processing anyway.
         
    # JPEG Normalization Array (m)
    m = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=float) * quality
    
    # Wait, simple list in numpy array definition might be error prone without commas if multi-line string used, 
    # but here I used list of lists.
    # Check line 42 in source: 49 64 78 ... 103 121 ...
    # My python line: [49, 64, 78, 87, 103 121, 120, 101]. 
    # Typo: 103 121 -> 103, 121.
    m = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=float) * quality
    
    # Zig-Zag Order (1-based in MATLAB)
    order_matlab = np.array([
        1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33,
        41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50,
        43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52,
        45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55,
        62, 63, 56, 64
    ])
    order = order_matlab - 1 # 0-based
    
    xm, xn = f.shape
    
    # Level shift
    f_shifted = f.astype(float) - 2**(round(bits) - 1)
    
    # Compute DCT matrix
    # T = tmat4e('DCT', 8)
    T = tmat4e('DCT', 8)
    
    # Block Processing
    # Need to process 8x8 blocks.
    # Pad if not multiple of 8? MATLAB 'blkproc' usually handles partials or expects padding?
    # The Code source doesn't show padding. We assume input is multiple of 8 or blkproc does something.
    # We will assume multiples of 8 or exact size handling.
    # If not multiple, we should pad locally.
    
    pad_h = (8 - xm % 8) % 8
    pad_w = (8 - xn % 8) % 8
    if pad_h > 0 or pad_w > 0:
        f_shifted = np.pad(f_shifted, ((0, pad_h), (0, pad_w)), 'constant')
        
    xm_pad, xn_pad = f_shifted.shape
    
    # View as blocks (creates logic similar to im2col/blkproc)
    # But we need "distinct" blocks.
    # Manual reshaping for efficiency.
    # Reshape to (M//8, 8, N//8, 8) -> Swap axes -> (M//8, N//8, 8, 8)
    
    n_blocks_row = xm_pad // 8
    n_blocks_col = xn_pad // 8
    
    blocks = f_shifted.reshape(n_blocks_row, 8, n_blocks_col, 8).transpose(0, 2, 1, 3)
    # Shape (n_rows, n_cols, 8, 8)
    
    # Apply DCT: T * block * T'
    # T is 8x8. block is ...8x8
    # We can broadcast matmul.
    # blocks: (N_blocks, 8, 8) where N_blocks = n_rows*n_cols
    blocks_flat = blocks.reshape(-1, 8, 8)
    
    dct_coeffs = T @ blocks_flat @ T.T
    # Or use scipy.fftpack.dct(..., norm='ortho')?
    # tmat4e gives orthonormal matrix, so manual matmul is correct equivalent to MATLAB.
    
    # Quantize: round(x ./ m)
    # m is 8x8. Broadcasts.
    q_coeffs = np.round(dct_coeffs / m)
    
    # Reorder (Zig-Zag)
    # We need to flatten each 8x8 block into 64 elements using 'order'.
    # Reshape each block to 64
    # Note: MATLAB `c = im2col(c, [8 8], 'distinct')` puts blocks into columns.
    # Then `c = c(order, :)` reorders the rows of that big matrix.
    # Our `blocks_flat` is (NumBlocks, 8, 8).
    # We want (64, NumBlocks) to match MATLAB column logic?
    # Or just (NumBlocks, 64) and reorder inner.
    
    # MATLAB: 
    # c = im2col(...) -> 64 x NumBlocks.
    # c = c(order, :) -> Reorders rows.
    # So 1st row of output is 1st ZigZag coeff of all blocks.
    
    # Let's flatten blocks properly.
    # blocks_flat is (NumBlocks, 8, 8).
    # We want to iterate block, reshape to 64 linear (row-major? or col-major?)
    # im2col 'distinct':
    # MATLAB stores matrices as column-major.
    # If we have 8x8 block in MATLAB: A.
    # A(:) is column vector [A(1,1), A(2,1), ...].
    # So `order` indices refer to this column-major flattening.
    # IN PYTHON/NUMPY, default is row-major.
    # So `block.flatten()` is [A(0,0), A(0,1)...].
    # We must be careful!
    
    # ZigZag Matrix in MATLAB `m` definition:
    # 1  2  6  7 ...
    # 3  5  8 ...
    # 4  9 ...
    # This visual pattern corresponds to indices.
    
    # If we look at `order`.
    # 1 (0,0), 9 (1,0) [col 1 start], 2 (0,1)?
    # Wait. MATLAB A(9) is A(1,2). (Row 1, Col 2).
    # Let's verify standard Zig-Zag.
    # (0,0) -> (0,1) -> (1,0) -> (2,0) -> ...
    # Python indices:
    # (0,0), (0,1), (1,0), (2,0), (1,1), (0,2)...
    
    # MATLAB Indices (Col-Major):
    # (1,1)=1.
    # (1,2)=9 (if 8 rows).
    # (2,1)=2.
    
    # `order` starts: 1, 9, 2...
    # 1 -> (1,1) -> Python (0,0).
    # 9 -> (1,2) -> Python (0,1).
    # 2 -> (2,1) -> Python (1,0).
    # 3 -> (3,1)? No.
    # Standard ZigZag usually: 0, 1, 5, 6, 14... for indices in linear layout?
    # No, visually:
    # 0, 1
    # 2, 3
    # Paths: (0,0) -> (0,1) -> (1,0) -> (2,0) -> (1,1) -> (0,2)...
    
    # Using `order` from MATLAB text is safest IF we respect MATLAB col-major indexing.
    # So, for each block `b` (8x8):
    # 1. Flatten it column-major: `b_col = b.flatten('F')`.
    # 2. Apply `order` (converted to 0-based): `b_ordered = b_col[order]`.
    
    # Let's do this for all blocks at once.
    # blocks_flat: (NumBlocks, 8, 8).
    # Transpose to (NumBlocks, 8, 8) -> We treat as collection of 8x8s.
    # We want: (64, NumBlocks) where each col is `b.flatten('F')`.
    
    # blocks_flat is (NumBlocks, 8, 8).
    # We want to flatten q_coeffs (DCT) not blocks_flat (Spatial).
    blocks_working = q_coeffs.transpose(0, 2, 1).reshape(-1, 64).T
    # Now rows are 0..63 pixel indices (Col-Major sense), Cols are blocks.
    
    # Reorder
    # c = c(order, :)
    c_ordered = blocks_working[order, :]
    
    # Run-length Encoding
    # MATLAB:
    # eob = max(c) + 1
    # for j=1:xb (blocks)
    #   i = last non-zero
    #   append c(1:i, j)
    #   append eob
    
    eob = np.max(c_ordered[:]) + 1
    r_list = []
    
    num_blocks = c_ordered.shape[1]
    
    for j in range(num_blocks):
        col = c_ordered[:, j]
        # Find last non-zero
        nz_indices = np.nonzero(col)[0]
        if len(nz_indices) == 0:
            last_nz = -1 # Empty block, only EOB? MATLAB: i=0. means c(1:0) -> empty. append eob.
        else:
            last_nz = nz_indices[-1]
            
        # Append elements up to last_nz
        # Python: 0 to last_nz inclusive -> slice :last_nz+1
        r_list.extend(col[:last_nz+1])
        r_list.append(eob)
        
    r = np.array(r_list)
    
    
    # Reshape r to column vector for mat2huff
    r = r.reshape(-1, 1)
    
    # Huffman Encode
    huff_struct = mat2huff(r)
    
    c_struct = {}
    c_struct['size'] = [xm, xn] # Original size
    c_struct['bits'] = bits
    c_struct['numblocks'] = num_blocks
    c_struct['quality'] = quality * 100
    c_struct['huffman'] = huff_struct
    
    return c_struct
