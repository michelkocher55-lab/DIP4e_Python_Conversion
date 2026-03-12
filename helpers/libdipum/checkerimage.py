
import numpy as np
from helpers.libdipum.isodd import isodd

def checkerimage(delX, M):
    """
    Generates a checkerboard image.
    
    Parameters:
    delX (float): Width of squares (inverse of). 
                  Squares are 1/delX pixels wide? 
                  Original doc: "squares are 1/delX pixels wide" is strange phrasing if delX=150.
                  Usually delX is step size in sampling space.
                  If delX << 1, we sample finely.
                  If delX > 1, we sample explicitly.
                  Actually doc says: "checkerboard image... sampling the checkerboard every delX pixels".
                  So delX is sampling period in 'checkerboard coordinate space' where squares are size 1.
    M (int): Size of image (MxM).
    
    Returns:
    g (ndarray): MxM checkerboard image (0 and 1).
    """
    
    # Generate sampling intervals
    # kx = 1 - delX
    # for I = 1:M
    #    kx = kx + delX + eps...
    
    # Vectorized generation of kx
    # kx starts at 1. Then steps by delX.
    # values at I (1..M): 1 + (I-1)*delX?
    # MATLAB loop:
    # I=1: kx = (1-delX) + delX = 1.
    # I=2: kx = 1 + delX.
    # so kx(I) = 1 + (I-1)*delX.
    
    # Plus EPS! MATLAB uses 'eps' (2.22e-16).
    # To handle precision issues at boundaries.
    
    indices = np.arange(M) # 0 to M-1
    kx_vals = 1.0 + indices * delX + np.finfo(float).eps * (indices + 1) # Mimic accumulation of eps? 
    # MATLAB loop adds eps every iteration. So eps accumulates?
    # kx = kx + delX + eps
    # So yes, kx_I = 1 + (I-1)*delX + I*eps?
    # Actually accumulating float addition errors usually key.
    # Let's perform exact loop logic to be safe or use simple vector with care.
    # Vector: kx = 1 + indices * (delX + eps) ? No, eps is constant.
    # Actually adding machine epsilon to delX effectively changes delX slightly.
    # But usually this is to ensure 1.0 is treated as 1.0+eps (in interval [1, 2)).
    
    # Let's trust vectorized approach:
    # kx = 1 + indices * delX
    # We add a small epsilon to avoid sitting exactly on integer boundary if delX divides exactly?
    # Let's add standard epsilon.
    kx_vals = 1.0 + indices * delX + np.finfo(float).eps
    
    # Calculate row vector r
    # if isodd(floor(kx)): val=0 else val=1
    # MATLAB returns 1 initially (white). If odd -> 0 (black).
    # isodd(floor(kx)) -> True (Odd integer) -> 0.
    # False (Even integer) -> 1.
    
    # so r = 0 if floor(kx) is odd, 1 if even.
    # r = 1 - (floor(kx) % 2)
    # Using isodd logic: 
    # odd_mask = isodd(np.floor(kx_vals))
    # r[odd_mask] = 0
    # r[~odd_mask] = 1
    
    floored_kx = np.floor(kx_vals)
    odd_mask = isodd(floored_kx)
    
    r = np.ones(M, dtype=int)
    r[odd_mask] = 0
    
    # c = r (Symmetric assumption)
    c = r
    
    # g(I, J) calculation
    # XOR logic derived earlier:
    # r=0, c=0 -> 0
    # r=1, c=0 -> 1
    # r=0, c=1 -> 1
    # r=1, c=1 -> 0
    # This is XOR.
    # numpy bitwise_xor
    
    # Outer product equivalent construction
    # g = r[:, None] ^ c[None, :]
    
    g = np.bitwise_xor(r[:, np.newaxis], c[np.newaxis, :]).astype(int)
    
    # MATLAB final values are 0 or 1.
    # Note: MATLAB code sets 1 initially. Then overwrites.
    # Logic matches XOR.
    
    return g
