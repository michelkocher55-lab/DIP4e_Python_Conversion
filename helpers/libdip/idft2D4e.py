import numpy as np

from helpers.libdip.dft2D4e import dft2D4e

def idft2D4e(F):
    """
    Computes the 2D inverse DFT.
    Uses a forward transform to compute the inverse.
    
    Parameters:
    -----------
    F : numpy.ndarray (complex)
        Frequency domain representation (M x N).
        
    Returns:
    --------
    f : numpy.ndarray (complex)
        Spatial domain image.
    """
    F = np.array(F)
    M, N = F.shape
    
    # Work with complex conjugate (Eq 4-158 in DIP4E)
    # The algorithm relies on: f = conj( DFT(conj(F)) ) / MN
    # The MATLAB implementation provided is:
    # F = (1/(M*N))*conj(F);
    # f = dft2D4e(F);
    # This effectively computes DFT(conj(F)/MN).
    # If standard properties hold, DFT(conj(F)) = conj(MN * f).
    # So this returns conj(f).
    # For real images, conj(f) == f.
    # We strictly follow the provided MATLAB code.
    
    F_conj = (1.0 / (M * N)) * np.conj(F)
    f = dft2D4e(F_conj)
    
    # Note: If f is expected to be real, this is sufficient.
    # If the original input to DFT was real, the output here is real (imag parts ~0).
    return f
