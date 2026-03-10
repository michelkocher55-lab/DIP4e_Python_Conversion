import numpy as np

def quantize4e(f, type_str='UNIFORM', bits=4):
    """
    Quantizes image f to specified number of bits.
    
    g = quantize4e(f, type_str, bits)
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image (uint8).
    type_str : str, optional
        'UNIFORM' (default): Zero the LSBs.
        'IGS': Improved Grayscale Quantization (Error Diffusion).
    bits : int, optional
        Number of bits to keep (1 to 8). Default is 4.
        
    Returns:
    --------
    g : numpy.ndarray
        Quantized image.
    """
    
    if f.dtype != np.uint8:
        # Warn or convert?
        # Assuming input is uint8.
        pass
        
    f = f.astype(np.uint8)
    g = f.copy()
    rows, cols = f.shape
    
    # Create masks
    # "bits" is number of MSBs to keep.
    # shift is how many LSBs to zero out.
    shift = 8 - bits
    
    # topMask: The bits we keep. e.g. bits=4 -> 11110000 (0xF0)
    top_mask = (0xFF << shift) & 0xFF
    
    # bottomMask: The bits we discard/diffuse. e.g. bits=4 -> 00001111 (0x0F)
    bottom_mask = (~top_mask) & 0xFF
    
    if type_str.upper() == 'UNIFORM':
        g = np.bitwise_and(f, top_mask)
        
    elif type_str.upper() == 'IGS':
        # Iterate over pixels
        # IGS is sequential error diffusion
        
        # We need to process row by row
        for r in range(rows):
            s = 0.0 # Sum (error accumulator), reset at start of line
            for c in range(cols):
                val = f[r, c]
                
                # Check for overflow risk in high bits
                # If the high bits are all 1, adding error might overflow
                if (val & top_mask) == top_mask:
                    s = float(val)
                else:
                    # Add current value + error from previous (lower bits of previous sum)
                    # previous sum's lower bits are s_prev & bottom_mask
                    # Wait, MATLAB code:
                    # sum = double(g(x,y)) + double(bitand(uint8(sum), bottomMask));
                    # Note: 'sum' is the variable name.
                    # bitand(uint8(s), bottomMask) gives the error part from previous
                    
                    prev_error = int(s) & bottom_mask
                    s = float(val) + float(prev_error)
                    
                # Store quantized result
                # g(x,y) = bitand(uint8(sum), topMask)
                res = int(s) & top_mask
                g[r, c] = res
                
    else:
        raise ValueError(f"Unknown type: {type_str}. Use 'UNIFORM' or 'IGS'.")
        
    return g
