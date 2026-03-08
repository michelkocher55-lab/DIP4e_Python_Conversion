import numpy as np

def fourier_descriptors(b):
    """
    Computes Fourier descriptors of a boundary.

    Args:
        b: A numpy array of shape (N, 2) containing ordered coordinates defining a boundary.

    Returns:
        z: A numpy array of complex Fourier descriptors.
    """
    b = np.array(b, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("b must be of size N-by-2.")

    np_points = b.shape[0]

    # If np is not even, force it to be by duplicating the end point.
    if np_points % 2 != 0:
        b = np.vstack([b, b[-1]])
        np_points += 1

    # Create an alternating sequence of 1s and -1s for use in centering the transform.
    x = np.arange(np_points)
    minusone = (-1) ** x
    
    # Multiply the input sequence by alternating 1s and -1s to center the transform.
    # Note: MATLAB code used b(:,1) = minusone.*b(:,1). Here we do it element-wise.
    b[:, 0] = minusone * b[:, 0]
    b[:, 1] = minusone * b[:, 1]

    # Convert coordinates to complex numbers.
    b_complex = b[:, 0] + 1j * b[:, 1]

    # Compute the descriptors.
    z = np.fft.fft(b_complex)
    
    return z
