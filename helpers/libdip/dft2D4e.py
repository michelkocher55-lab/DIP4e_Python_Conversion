import numpy as np

def dft2D4e(f):
    """
    Computes the 2D forward FFT.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image/matrix.
        
    Returns:
    --------
    F : numpy.ndarray (complex)
        The 2D DFT of the input.
    """
    # Convert f to array (preserve complex if present)
    f = np.array(f)
    if np.iscomplexobj(f):
         pass # Already complex
    else:
         f = f.astype(float)
    
    # Compute 2D FFT
    # Equivalent to fft(fft(f, [], 2), [], 1) in MATLAB
    F = np.fft.fft2(f)
    
    return F
