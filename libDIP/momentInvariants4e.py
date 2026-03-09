from typing import Any
import numpy as np


def momentInvariants4e(F: Any):
    """
    Compute moment invariant of an image.

    Parameters:
    -----------
    F : numpy.ndarray
        Input image (2-D).

    Returns:
    --------
    phi : list
        Seven-element list containing the Hu moment invariants.
    """
    F = F.astype(float)

    # Compute Raw Moments
    m = compute_m(F)

    # Compute Normalized Central Moments
    eta = compute_eta(m)

    # Compute Hu Invariants
    phi = compute_phi(eta)

    return phi


def compute_m(F: Any):
    """Computes raw moments up to order 3."""
    M, N = F.shape

    # Meshgrid matches MATLAB's meshgrid(1:N, 1:M)
    # x (cols), y (rows)
    x, y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))

    # Flatten for easier sum
    x = x.flatten()
    y = y.flatten()
    F_flat = F.flatten()

    m = {}

    # m00
    m["m00"] = np.sum(F_flat)
    if m["m00"] == 0:
        m["m00"] = np.finfo(float).eps

    # Central moments components
    # Just raw moments here
    m["m10"] = np.sum(x * F_flat)
    m["m01"] = np.sum(y * F_flat)
    m["m11"] = np.sum(x * y * F_flat)
    m["m20"] = np.sum((x**2) * F_flat)
    m["m02"] = np.sum((y**2) * F_flat)
    m["m30"] = np.sum((x**3) * F_flat)
    m["m03"] = np.sum((y**3) * F_flat)
    m["m12"] = np.sum(x * (y**2) * F_flat)
    m["m21"] = np.sum((x**2) * y * F_flat)

    return m


def compute_eta(m: Any):
    """Computes normalized central moments."""
    m00 = m["m00"]
    xbar = m["m10"] / m00
    ybar = m["m01"] / m00

    eta = {}

    # Helper to normalize
    # eta_pq = mu_pq / m00^gamma
    # gamma = (p+q)/2 + 1

    # Calculate Central Moments mu_pq first?
    # MATLAB formulas directly compute eta using raw moments and centroid.

    # eta11 = (m11 - ybar*m10) / m00^2
    eta["eta11"] = (m["m11"] - ybar * m["m10"]) / (m00**2)

    # eta20 = (m20 - xbar*m10) / m00^2
    eta["eta20"] = (m["m20"] - xbar * m["m10"]) / (m00**2)

    # eta02 = (m02 - ybar*m01) / m00^2
    eta["eta02"] = (m["m02"] - ybar * m["m01"]) / (m00**2)

    # eta30
    eta["eta30"] = (m["m30"] - 3 * xbar * m["m20"] + 2 * (xbar**2) * m["m10"]) / (
        m00**2.5
    )

    # eta03
    eta["eta03"] = (m["m03"] - 3 * ybar * m["m02"] + 2 * (ybar**2) * m["m01"]) / (
        m00**2.5
    )

    # eta21
    # MATLAB: (m.m21 - 2 * xbar * m.m11 - ybar * m.m20 + 2 * xbar^2 * m.m01)
    eta["eta21"] = (
        m["m21"] - 2 * xbar * m["m11"] - ybar * m["m20"] + 2 * (xbar**2) * m["m01"]
    ) / (m00**2.5)

    # eta12
    # MATLAB: (m.m12 - 2 * ybar * m.m11 - xbar * m.m02 + 2 * ybar^2 * m.m10)
    eta["eta12"] = (
        m["m12"] - 2 * ybar * m["m11"] - xbar * m["m02"] + 2 * (ybar**2) * m["m10"]
    ) / (m00**2.5)

    return eta


def compute_phi(e: Any):
    """Computes 7 Hu moment invariants."""
    phi = [0.0] * 7

    # phi1 = eta20 + eta02
    phi[0] = e["eta20"] + e["eta02"]

    # phi2 = (eta20 - eta02)^2 + 4*eta11^2
    phi[1] = (e["eta20"] - e["eta02"]) ** 2 + 4 * (e["eta11"] ** 2)

    # phi3 = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2
    phi[2] = (e["eta30"] - 3 * e["eta12"]) ** 2 + (3 * e["eta21"] - e["eta03"]) ** 2

    # phi4 = (eta30 + eta12)^2 + (eta21 + eta03)^2
    phi[3] = (e["eta30"] + e["eta12"]) ** 2 + (e["eta21"] + e["eta03"]) ** 2

    # phi5
    # (eta30 - 3*eta12) * (eta30 + eta12) * [ (eta30 + eta12)^2 - 3*(eta21 + eta03)^2 ] + ...
    # (3*eta21 - eta03) * (eta21 + eta03) * [ 3*(eta30 + eta12)^2 - (eta21 + eta03)^2 ]

    t1 = e["eta30"] + e["eta12"]
    t2 = e["eta21"] + e["eta03"]

    phi[4] = (e["eta30"] - 3 * e["eta12"]) * t1 * (t1**2 - 3 * t2**2) + (
        3 * e["eta21"] - e["eta03"]
    ) * t2 * (3 * t1**2 - t2**2)

    # phi6
    # (eta20 - eta02) * [ t1^2 - t2^2 ] + 4*eta11 * t1 * t2
    phi[5] = (e["eta20"] - e["eta02"]) * (t1**2 - t2**2) + 4 * e["eta11"] * t1 * t2

    # phi7
    # (3*eta21 - eta03) * t1 * [ t1^2 - 3*t2^2 ] + ...
    # (3*eta12 - eta30) * t2 * [ 3*t1^2 - t2^2 ] -- Wait, MATLAB says:
    # (3*e.eta12 - e.eta30) * (e.eta21 + e.eta03) * ...
    # Correct.

    phi[6] = (3 * e["eta21"] - e["eta03"]) * t1 * (t1**2 - 3 * t2**2) + (
        3 * e["eta12"] - e["eta30"]
    ) * t2 * (
        3 * t1**2 - t2**2
    )  # Note 3*eta12 - eta30 is Negative of phi5's 1st term's (eta30 - 3*eta12)

    return phi
