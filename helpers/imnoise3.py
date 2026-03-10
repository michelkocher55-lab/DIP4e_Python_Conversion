from typing import Any
import numpy as np


def imnoise3(M: Any, N: Any, C: Any, A: Any = None, B: Any = None):
    """
    Generates a spatial sinusoidal pattern.

    [r, F, S] = imnoise3(M, N, C, A, B)

    Parameters:
    M, N: Size of output pattern
    C: Kx2 matrix of frequency domain coordinates (u, v) for K impulses.
       Coordinates are integers w.r.t center of frequency rectangle.
    A: 1xK vector of amplitudes (default: all 1s)
    B: Kx2 matrix of phase components (default: all 0s)

    Returns:
    r: Spatial sinusoidal pattern (real part of inverse transform)
    F: Fourier transform (centered)
    S: Spectrum (abs(F))
    """

    C = np.array(C)
    if C.ndim == 1:
        C = C[np.newaxis, :]

    K = C.shape[0]

    # Defaults
    if A is None:
        A = np.ones(K)
    elif len(A) == 0:  # Empty list or array passed explicitly checks
        A = np.ones(K)
    else:
        A = np.array(A).flatten()

    if B is None:
        B = np.zeros((K, 2))
    else:
        B = np.array(B)

    # Validation
    # MATLAB: floor(M/2)
    limit_u = M // 2
    limit_v = N // 2

    if np.any(np.abs(C[:, 0]) > limit_u) or np.any(np.abs(C[:, 1]) > limit_v):
        raise ValueError("Impulses must be inside the frequency rectangle.")

    if np.any(np.floor(C) != C):
        raise ValueError("Impulse coordinates must be integers.")
    C = C.astype(int)

    # Center
    # MATLAB: floor(M/2) + 1. (1-based index)
    # Python: M // 2. (0-based index)
    # Example M=5. MATLAB center=3. Python center=2.
    ucenter = M // 2
    vcenter = N // 2

    F = np.zeros((M, N), dtype=complex)

    for k in range(K):
        # Coordinates
        u1 = ucenter + C[k, 0]
        v1 = vcenter + C[k, 1]

        u2 = ucenter - C[k, 0]
        v2 = vcenter - C[k, 1]

        # MATLAB indices u1, v1 are used in phase formula "u1*B/M".
        # If we use Python indices u1, v1, they are 1 less than MATLAB indices.
        # This introduces a constant phase shift which corresponds to a spatial translation.
        # However, for pure noise generation usually exact phase w.r.t origin definition
        # (corner vs center) is tricky.
        # Let's use Python indices directly. If precise numerical equivalence to MATLAB
        # regarding phase origin is required, we might need to add 1.
        # But standard FFT definitions usually use 0-based indexing.
        # MATLAB's 1-based indexing in the exponential (u1 * ...) might actually be an artifact
        # of using array indices as coordinates?
        # Actually Eq 5-9 in DIP4E likely uses $u$ (frequency variable).
        # But the code uses `u1`. `u1` corresponds to centered coordinate + center_offset.
        # It approximates the uncentered index $0..M-1$.
        # Let's stick to Python indices.

        # Eq terms
        # Impulse 1
        # magnitude: 1j * M * N * (A[k]/2)
        # phase: -1j * 2 * pi * (u1 * B[k,0]/M + v1 * B[k,1]/N)
        term1 = 1j * M * N * (A[k] / 2.0)
        phase1 = -1j * 2 * np.pi * (u1 * B[k, 0] / M + v1 * B[k, 1] / N)
        val1 = term1 * np.exp(phase1)

        # Impulse 2 (conjugate)
        # magnitude: -1j * M * N * (A[k]/2)
        # phase: 1j * 2 * pi * (u2 * B[k,0]/M + v2 * B[k,1]/N)
        term2 = -1j * M * N * (A[k] / 2.0)
        phase2 = 1j * 2 * np.pi * (u2 * B[k, 0] / M + v2 * B[k, 1] / N)
        val2 = term2 * np.exp(phase2)

        F[u1, v1] = val1
        F[u2, v2] = val2

    S = np.abs(F)

    # Compute spatial pattern
    # F is centered. ifft2 expects uncentered (DC at 0,0).
    # MATLAB: ifft2(ifftshift(F)).
    # ifftshift moves center to corners.

    r = np.real(np.fft.ifft2(np.fft.ifftshift(F)))

    return r, F, S
