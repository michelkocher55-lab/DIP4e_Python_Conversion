
import numpy as np
from numpy.linalg import norm

def ReconstructionUsingBitPlanes(f, BitPlanes, NBits, NBitsRec):
    """
    Reconstructs image using the N most significant bit planes.
    
    Parameters:
    f: Original image (for SNR calc).
    BitPlanes: List of bit planes. BitPlanes[0] corresponds to MSB.
    NBits: Total bits (e.g. 8).
    NBitsRec: Number of planes to use (starting from MSB).
    
    Returns:
    Rec: Reconstructed image.
    SNR: Signal to Noise Ratio in dB.
    """
    # Rec = zeros (size (f), 'uint8');
    # MATLAB: BitPlanes(:, :, iter) where iter=1 is MSB (NBits-1 power).
    
    f = f.astype(float)
    Rec = np.zeros_like(f)
    
    for iter_idx in range(NBitsRec): # 0 to NBitsRec-1
        # MATLAB: iter goes 1 to NBitsRec
        # code: Rec = Rec + uint8(BitPlanes(:,:,iter)) * 2^(NBits-iter)
        # i=1: 2^(8-1) = 128 (MSB)
        
        # Python: idx goes 0 to NBitsRec-1.
        # Assuming BitPlanes input list is ordered [MSB, ..., LSB]
        # plane corresponding to iter_idx=0 is MSB.
        
        # Power: NBits - (iter_idx + 1) for 1-based logic equivalent
        # If NBits=8, iter=0. Pow = 7. 2^7=128. Correct.
        
        power_val = 2**(NBits - (iter_idx + 1))
        Rec += BitPlanes[iter_idx] * power_val
        
    Rec = Rec.astype(np.uint8)
    
    # SNR = 20 * log10 (norm (double(f(:))) / norm(double(f(:)) - double(Rec(:))));
    f_double = f.flatten()
    Rec_double = Rec.astype(float).flatten()
    
    diff_norm = norm(f_double - Rec_double)
    
    if diff_norm == 0:
        SNR = float('inf')
    else:
        SNR = 20 * np.log10(norm(f_double) / diff_norm)
        
    return Rec, SNR
