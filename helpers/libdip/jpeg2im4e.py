import numpy as np

from helpers.libdipum.huff2mat import huff2mat
from helpers.libdip.tmat4e import tmat4e

def jpeg2im4e(c):
    """
    Decodes an IM2JPEG4E compressed image.
    
    Parameters:
    -----------
    c : dict
        Compression structure from im2jpeg4e.
        
    Returns:
    --------
    g : numpy.ndarray
        Reconstructed image.
    """
    # Parameters
    quality_scale = float(c['quality']) / 100.0
    bits = int(c['bits'])
    xb = int(c['numblocks'])
    xm, xn = c['size']
    
    # Normalizing Array (m) - Same as encoder
    m = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=float) * quality_scale
    
    # Scan Order
    order_matlab = np.array([
        1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33,
        41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50,
        43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52,
        45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55,
        62, 63, 56, 64
    ])
    order = order_matlab - 1
    
    # Compute inverse ordering
    # MATLAB: rev(k) = find(order == k) (1-based)
    # Python: rev is argsort of order? 
    # If order[k] maps input index k to output index order[k].
    # We want to map back.
    # z[order] vs z[rev].
    # Encoder: c_ordered = raw[order].
    # Decoder needs: raw = c_ordered[rev].
    # Let's useargsort.
    rev = np.argsort(order)
    
    # Huffman Decode
    g_raw = huff2mat(c['huffman'])
    
    # EOB
    if len(g_raw) == 0:
        # Handle empty?
        pass
    eob = np.max(g_raw)
    
    # Reconstruct columns from run-length
    # z = zeros(64, xb)
    z = np.zeros((64, xb))
    k = 0 # index in g_raw
    
    # Optimization: Iterating python loop for pixels is slow?
    # But Run-length decoding is inherently serial unless we have markers.
    # We iterate blocks.
    
    # Flatten g_raw for easy indexing
    g_flat = g_raw.flatten()
    
    for j in range(xb):
        # Read until EOB
        # Optimized search?
        # Ideally we'd find all EOB indices first?
        # But let's follow logic.
        
        # Fill column j
        i = 0
        while i < 64 and k < len(g_flat):
            val = g_flat[k]
            k += 1
            if val == eob:
                break
            z[i, j] = val
            i += 1
            
    # Restore Order
    # z was in ZigZag order (rows).
    # We apply 'rev' to shuffle rows back to linear-column-major order.
    z = z[rev, :]
    
    # Now z cols are blocks flattened by column ('F').
    # We need to reshape columns back to 8x8 blocks.
    # z[:, j] -> (8,8) column major.
    
    # Vectorized Reshape
    # z is (64, xb).
    # Transpose to (xb, 64). reshape to (xb, 8, 8) but respecting 'F'.
    # A flat array 'A' from MATLAB 'F' order of 8x8:
    # A = [c1, c2, ... c8].
    # In Python, we can reshape (8, 8) with order='F'.
    # But we have multiple blocks.
    
    # Let's do:
    # 1. z.T -> (xb, 64).
    # 2. Reshape each 64-vector to 8x8 'F'.
    
    # z.T is row-major of (xb, 64). Elements are correct sequence for 'F' filling.
    # If we reshape (xb, 8, 8), standard numpy fills last dim first.
    # (row 0, col 0), (row 0, col 1)... -> This is Row Major.
    # BUT we want Column Major.
    # So we should reshape to (xb, 8, 8) but interpreting data as Col Major?
    # No, we construct (xb, 8, 8) such that reading it gives 'z'.
    # Actually, simpler:
    # z column j is [ (0,0), (1,0), (2,0)... (7,0), (0,1)... ]
    # This is exactly 'F' order.
    # So we reshape z.T to (xb, 8, 8) using order='F'? No, z.T mixes blocks.
    
    # Let's reconstruct (NumBlocks, 8, 8)
    blocks = np.zeros((xb, 8, 8))
    for j in range(xb):
        blocks[j, :, :] = z[:, j].reshape((8, 8), order='F')
        
    # De-Normalize (De-Quantize)
    # block = block .* m
    # m is 8x8. Broadcast.
    blocks = blocks * m
    
    # Inverse DCT
    # T' * block * T
    T = tmat4e('DCT', 8)
    
    # blocks: (xb, 8, 8)
    # idct_blocks = T.T @ blocks @ T
    idct_blocks = T.T @ blocks @ T
    
    # Level Shift
    idct_blocks += 2**(bits - 1)
    
    # Cast
    if bits <= 8:
        g = np.clip(idct_blocks, 0, 255).astype(np.uint8)
    else:
        g = np.clip(idct_blocks, 0, 65535).astype(np.uint16)
        
    # Col2Im (Reassemble image)
    # We have list of blocks. Need to arrange into (xm, xn)
    # Original order was row-wise blocks usually?
    # Encoder:
    # blocks = f_shifted.reshape(n_blocks_row, 8, n_blocks_col, 8).transpose(0, 2, 1, 3)
    # ...
    # blocks_flat = blocks.reshape(-1, 8, 8).
    # So blocks are ordered by row, then col. (Row 0 blocks, then Row 1 blocks...)
    # Wait, reshape(n_rows, 8, n_cols, 8).
    # Then transpose(0, 2, ...). -> (n_rows, n_cols, 8, 8).
    # Then reshape(-1, 8, 8).
    # Default reshape is Row-Major (C).
    # So index 0 is (r0, c0). Index 1 is (r0, c1).
    # So yes, row-by-row of blocks.
    
    # Need to reconstruct (n_rows, n_cols, 8, 8)
    # Padded dimensions!
    pad_h = (8 - xm % 8) % 8
    pad_w = (8 - xn % 8) % 8
    xm_pad = xm + pad_h
    xn_pad = xn + pad_w
    
    n_rows = xm_pad // 8
    n_cols = xn_pad // 8
    
    rebuilt_blocks = g.reshape(n_rows, n_cols, 8, 8)
    
    # Convert to image (xm_pad, xn_pad)
    # (n_rows, n_cols, 8, 8) -> (n_rows, 8, n_cols, 8) -> reshape
    rebuilt_img = rebuilt_blocks.transpose(0, 2, 1, 3).reshape(xm_pad, xn_pad)
    
    # Crop padding
    rebuilt_img = rebuilt_img[:xm, :xn]
    
    return rebuilt_img
