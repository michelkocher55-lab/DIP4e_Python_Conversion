from typing import Any
import numpy as np


def levelSetReInit4e(phi: Any, niter: Any = 5, delT: Any = 0.5):
    """
    Reinitializes signed distance function.

    phinew = levelsetReinit4e(phi, niter=5, delT=0.5)

    Parameters
    ----------
    phi : numpy.ndarray
        Input level set function.
    niter : int
        Number of iterations.
    delT : float
        Time step.

    Returns
    -------
    phinew : numpy.ndarray
        Reinitialized signed distance function (|grad(phi)| = 1).
    """

    phi = np.array(phi, dtype=float)
    M, N = phi.shape
    phi0 = phi.copy()
    S = np.sign(phi0)

    # Smooth original function gradient (for S)
    # MATLAB: [phi0y, phi0x] = gradient(phi0) -> [Cols, Rows]
    # Python: gy, gx = gradient(phi0) -> [Rows, Cols]
    # So phi0y (MATLAB) corresponds to gy (Python, Row Deriv).
    # phi0x (MATLAB) corresponds to gx (Python, Col Deriv).
    # Norm is same.
    gy, gx = np.gradient(phi0)
    e = np.finfo(float).eps
    phi0 = phi0 / (
        np.sqrt(gx**2 + gy**2) + e
    )  # This step normalizes phi0? MATLAB line 35.
    # Actually MATLAB line 35 modifies phi0 itself? Yes.

    # Iterate
    for i in range(niter):
        G = computeG(phi, phi0)
        phinew = phi - delT * S * G

        # Stopping rule
        diff_sum = np.sum(np.abs(phinew - phi))
        if diff_sum < delT * max(M, N):
            phi = phinew
            break

        phi = phinew

    return phi


def computeG(phi: Any, phi0: Any):
    """
    Computes G matrix for reinitialization.
    based on Godunov's scheme / Sussman.
    """
    # Pad phi
    # MATLAB: [1 1], 'replicate', 'both'
    phi_pad = np.pad(phi, ((1, 1), (1, 1)), mode="edge")

    Mpad, Npad = phi_pad.shape

    # Indices for center (equivalent to i, j in MATLAB)
    # MATLAB i=2:Mpad-1 -> Python slice 1:-1

    # a: Backward difference in x (Rows? MATLAB label in comments says x-direction, but indices are (i,j) vs (i-1,j))
    # MATLAB: a = phi(i,j) - phi(i-1,j). i is Row index. So 'x-direction' here refers to ROWS.
    a = phi_pad[1:-1, 1:-1] - phi_pad[:-2, 1:-1]

    # b: Forward difference in x (Rows)
    # MATLAB: b = phi(i+1,j) - phi(i,j).
    b = phi_pad[2:, 1:-1] - phi_pad[1:-1, 1:-1]

    # c: Backward difference in y (Cols)
    # MATLAB: c = phi(i,j) - phi(i,j-1).
    c = phi_pad[1:-1, 1:-1] - phi_pad[1:-1, :-2]

    # d: Forward difference in y (Cols)
    # MATLAB: d = phi(i,j+1) - phi(i,j).
    d = phi_pad[1:-1, 2:] - phi_pad[1:-1, 1:-1]

    # Conditions
    aplus = np.maximum(a, 0)
    aminus = np.minimum(a, 0)
    bplus = np.maximum(b, 0)
    bminus = np.minimum(b, 0)
    cplus = np.maximum(c, 0)
    cminus = np.minimum(c, 0)
    dplus = np.maximum(d, 0)
    dminus = np.minimum(d, 0)

    Ap = phi0 > 0
    An = phi0 < 0

    # Gp
    # sqrt(max(aplus^2, bminus^2) + max(cplus^2, dminus^2)) - 1
    Gp = np.sqrt(np.maximum(aplus**2, bminus**2) + np.maximum(cplus**2, dminus**2)) - 1

    # Gn
    # sqrt(max(aminus^2, bplus^2) + max(cminus^2, dplus^2)) - 1
    Gn = np.sqrt(np.maximum(aminus**2, bplus**2) + np.maximum(cminus**2, dplus**2)) - 1

    G = Gp * Ap + Gn * An

    return G
