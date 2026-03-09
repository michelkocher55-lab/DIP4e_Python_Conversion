from typing import Any
import numpy as np
import numpy.fft as fft
# Note: In MATLAB implementation, it uses dft2D4e and minusOne4e.
# Here we can use standard numpy FFT but we must handle the centering logic
# consistent with how we expect H to be centered.


def constrainedLsTF4e(H: Any, gam: Any):
    """
    Constrained least squares filter transfer function.

    Parameters:
    -----------
    H : numpy.ndarray
        Degradation transfer function.
        Must be of even dimensions to match MATLAB logic for Laplacian symmetry,
        but we can support odd.
        However, to strictly match the requested function:
    gam : float
        Gamma parameter (regularization).

    Returns:
    --------
    L : numpy.ndarray
        Filter transfer function.
    """
    H = np.array(H)
    M, N = H.shape

    # MATLAB: Centers Laplacian.
    # p = zeros(M,N)
    # p(cx, cy) = 4, neighbors -1.
    # Center cx = floor(M/2)+1. (1-based)
    # Python 0-based: cx = M // 2.

    cx = M // 2
    cy = N // 2

    p = np.zeros((M, N))
    p[cx, cy] = 4
    p[cx - 1, cy] = -1
    p[cx + 1, cy] = -1
    p[cx, cy - 1] = -1
    p[cx, cy + 1] = -1

    # MATLAB: P = dft2D4e(minusOne4e(p))
    # It centers p first (multiply by -1^(x+y)), then FFT.
    # The 'minusOne4e' centering shifts the DC to corners before FFT,
    # or centers the frequency domain after FFT.

    # Wait, dft2D4e usually expects centered spatial data if we use minusOne4e.
    # DIP4E Logic:
    # "frequency domain centering" typically means shifting DC to center.
    # If p is spatial, simply taking fft2(p) gives P with DC at (0,0).
    # If we want P centered (DC at M/2, N/2), we use fftshift(fft2(p)).
    # MATLAB code: `P = dft2D4e(minusOne4e(p))`.
    # If dft2D4e is standard FFT, then `minusOne4e(p)` (spatial modulation)
    # results in a frequency domain shift.
    # Yes, multiplying by (-1)^(x+y) in spatial domain centers the frequency domain.
    # So P has DC at center.

    # Let's replicate this:
    # 1. Construct p (spatial Laplacian).
    # 2. Modulate p to center transform: p_cent = p * (-1)^(x+y)
    # 3. FFT.

    x = np.arange(M)
    y = np.arange(N)
    X_grid, Y_grid = np.meshgrid(x, y, indexing="ij")

    # minusOne: (-1)^(x+y)
    modulate = (-1) ** (X_grid + Y_grid)

    p_modulated = p * modulate

    # FFT
    P = fft.fft2(p_modulated)

    # Magnitude squared
    Pabs2 = np.abs(P) ** 2

    # Hstar
    Hstar = np.conj(H)
    Habs2 = np.abs(H) ** 2

    # L
    L = Hstar / (Habs2 + gam * Pabs2)

    return L
