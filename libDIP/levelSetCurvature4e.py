from typing import Any
import numpy as np


def levelSetCurvature4e(phi: Any, mode: Any = "Cu"):
    """
    Computes the curvature of a level set function.

    K = levelsetCurvature4e(phi, mode='Cu')

    Parameters
    ----------
    phi : numpy.ndarray
        Level set function.
    mode : str
        'Cu' (default): Unnormalized curvature.
        'Cn': Normalized by max amplitude.

    Returns
    -------
    K : numpy.ndarray
        Curvature field.
    """

    phi = np.array(phi, dtype=float)
    M, N = phi.shape

    # Pad with ones (replicating behavior of MATLAB imPad4e(phi,1,1,'constant','both',1))
    # Note: MATLAB code uses 1 for padding value?
    # Usually curvature boundary conditions are mirror/replicate.
    # But line 17 in MATLAB says `imPad4e(phi,1,1,'constant','both',1)`.
    # Let's stick to MATLAB logic: constant pad with value 1?
    # Or maybe it meant 'replicate'?
    # It seems to pad with 1s locally? No, 1 is the value.
    # Wait, if phi is SDF (distance), padding with 1 (small positive) implies "outside".
    phip = np.pad(phi, ((1, 1), (1, 1)), mode="edge")
    # Wait, MATLAB code line 37 explicitly compensates for border effects by replicating rows/cols.
    # So `mode='edge'` (replicate) is safer and likely what was intended or functionally better.
    # The MATLAB implementation lines 39-43 overwrite borders with neighbors anyway.

    # Central Differences
    # MATLAB: phip(3:end, 2:N+1) -> Python: phip[2:, 1:-1]
    # MATLAB: phip(1:M, 2:N+1)   -> Python: phip[:-2, 1:-1]

    # phix (Derivative along Rows? NO. phix usually means d/dx. In image processing x is usually Col, y is Row.
    # BUT let's check MATLAB code:
    # phix = 0.5*(phip(3:end,2:N+1) - phip(1:M,2:N+1));
    # This difference is along dimension 1 (Rows). So phix is dPhi/dRow.
    # phiy = 0.5*(phip(2:M+1,3:end) - phip(2:M+1,1:N));
    # This difference is along dimension 2 (Cols). So phiy is dPhi/dCol.

    # Python equivalent:
    # phix (Row diff): (phip[2:, ...] - phip[:-2, ...])
    phix = 0.5 * (phip[2:, 1:-1] - phip[:-2, 1:-1])
    phiy = 0.5 * (phip[1:-1, 2:] - phip[1:-1, :-2])

    phixx = phip[2:, 1:-1] + phip[:-2, 1:-1] - 2 * phi
    phiyy = phip[1:-1, 2:] + phip[1:-1, :-2] - 2 * phi

    # phixy = 0.25 * (phip(3:end,3:end) - phip(1:M,3:end) - phip(3:end,1:N) + phip(1:M,1:N))
    # Indices:
    # 3:end -> 2:
    # 1:M   -> :-2
    # 1:N   -> :-2
    phixy = 0.25 * (phip[2:, 2:] - phip[:-2, 2:] - phip[2:, :-2] + phip[:-2, :-2])

    # Compute Curvature
    # K = ... / DEN
    DEN = (phix**2 + phiy**2 + np.finfo(float).eps) ** 1.5
    K = (phixx * (phiy**2) - 2 * (phix * phiy * phixy) + phiyy * (phix**2)) / DEN

    # Normalize
    if mode == "Cn":
        K = K / (np.max(np.abs(K)) + np.finfo(float).eps)

    # Compensate borders (Replicate)
    K[:, 0] = K[:, 1]
    K[:, -1] = K[:, -2]
    K[0, :] = K[1, :]
    K[-1, :] = K[-2, :]

    return K
