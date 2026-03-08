import numpy as np

def fourier_descriptors_inverse(z, nd=None):
    """
    Computes inverse Fourier descriptors.

    Args:
        z: A sequence of Fourier descriptors.
        nd: The number of descriptors used to compute the inverse. 
            Must be an even integer no greater than len(z). 
            If omitted, defaults to len(z).

    Returns:
        b: A numpy array of shape (len(z), 2) containing the coordinates of a closed boundary.
    """
    # Ensure z is a numpy array and copy to avoid modifying the input
    z = np.array(z, dtype=complex).copy()
    np_points = len(z)
    
    if nd is None:
        nd = np_points
        
    if np_points % 2 != 0:
        raise ValueError("length(z) must be an even integer.")
    if nd % 2 != 0:
        raise ValueError("nd must be an even integer.")
        
    # Create an alternating sequence of 1s and -1s for use in uncentering the transform.
    x = np.arange(np_points)
    minusone = (-1.0) ** x
    
    # Use only nd descriptors in the inverse.
    d = (np_points - nd) // 2
    if d > 0:
        z[:d] = 0
        z[np_points-d:] = 0
        
    # Compute the inverse and convert back to spatial coordinates.
    # Note: ifft returns complex numbers, we take real and imag parts.
    zz = np.fft.ifft(z)
    
    b = np.zeros((np_points, 2))
    b[:, 0] = zz.real
    b[:, 1] = zz.imag
    
    # Multiply by alternating 1s and -1s to undo the centering.
    b[:, 0] = minusone * b[:, 0]
    b[:, 1] = minusone * b[:, 1]
    
    return b
